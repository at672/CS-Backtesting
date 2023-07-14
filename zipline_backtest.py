import pandas as pd
import pandas_datareader.data as web

import matplotlib.pyplot as plt
import numpy as np

from zipline import run_algorithm
#from zipline.api import order_target, record, symbol
from zipline.api import (
    attach_pipeline,
    date_rules,
    order_target,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    symbol,
    time_rules,
    get_open_orders,
)
from zipline.finance import commission, slippage
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import SimpleMovingAverage
from zipline.pipeline.data import USEquityPricing

import pyfolio as pf

import warnings
warnings.filterwarnings('ignore')

# FOR JUPYTER NOTEBOOKS
# %load_ext zipline


### Strategy

def initialize(context):
    context.symbols = [
        symbol("SPY"),
        symbol("EFA"),
        symbol("IEF"),
        symbol("VNQ"),
        symbol("GSG"),
    ]
    context.sma = {}
    context.period = 10 * 21

    for asset in context.symbols: 
        context.sma[asset] = SimpleMovingAverage(
            inputs=[USEquityPricing.close],
            window_length=context.period
        )

    schedule_function(
        func=rebalance,
        date_rule=date_rules.month_start(),
        time_rule=time_rules.market_open(minutes=1),
    )

    context.set_commission(
        commission.PerShare(cost=0.01, min_trade_cost=1.00)
    )
    context.set_slippage(slippage.VolumeShareSlippage())


def rebalance(context, data):
    
    longs = [
        asset
        for asset in context.symbols
        if data.current(asset, "price") > context.sma[asset].mean()
    ]

    for asset in context.portfolio.positions:
        if asset not in longs and data.can_trade(asset):
            order_target_percent(asset, 0)

    for asset in longs:
        if data.can_trade(asset):
            order_target_percent(asset, 1.0 / len(longs))


start = pd.Timestamp("2010")
end = pd.Timestamp("2023-06-30")

# test using datetime objects
# start = datetime.datetime(2015, 1, 1)
# end = datetime.datetime(2020, 1, 10)
# data = web.DataReader('AAPL', 'yahoo', start, end)

perf = run_algorithm(
    start=start,
    end=end,
    initialize=initialize,
    capital_base=100000,
    #bundle="quandl-eod"
    bundle = "quandl"
)

returns, positions, transactions = \
    pf.utils.extract_rets_pos_txn_from_zipline(perf)


pf.create_full_tear_sheet(
    returns,
    positions=positions,
    transactions=transactions,
    round_trips=True,
)