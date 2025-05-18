import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from uniswap_v3_backtester import UniswapV3Backtest

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Parameter sweep configurations
random_seed = 420
parameter_sets = [
    {
        "name": "baseline",
        "initial_capital": 100000,
        "range_width_percent": 0.2,
        "rebalance_threshold": 0.1,
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 4, 1),
        "eth_start_price": 2000,
        "drift": 0,
        "volatility": 0.0025,
        "random_seed": random_seed,
        "gas_price_range": (15, 55),
        "volume_range": (1e5, 2e5)
    },
    {
        "name": "high_volatility",
        "initial_capital": 100000,
        "range_width_percent": 0.2,
        "rebalance_threshold": 0.1,
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 4, 1),
        "eth_start_price": 2000,
        "drift": 0,
        "volatility": 0.015,
        "random_seed": random_seed,
        "gas_price_range": (15, 55),
        "volume_range": (1e5, 2e5)
    },
    {
        "name": "low_volatility",
        "initial_capital": 100000,
        "range_width_percent": 0.2,
        "rebalance_threshold": 0.1,
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 4, 1),
        "eth_start_price": 2000,
        "drift": 0,
        "volatility": 0.0015,
        "random_seed": random_seed,
        "gas_price_range": (15, 55),
        "volume_range": (1e5, 2e5)
    },
    {
        "name": "positive_drift",
        "initial_capital": 100000,
        "range_width_percent": 0.2,
        "rebalance_threshold": 0.1,
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 4, 1),
        "eth_start_price": 2000,
        "drift": 0.0005,
        "volatility": 0.0075,
        "random_seed": random_seed,
        "gas_price_range": (15, 55),
        "volume_range": (1e5, 2e5)
    },
    {
        "name": "negative_drift",
        "initial_capital": 100000,
        "range_width_percent": 0.2,
        "rebalance_threshold": 0.1,
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 4, 1),
        "eth_start_price": 2000,
        "drift": -0.0005,
        "volatility": 0.0075,
        "random_seed": random_seed,
        "gas_price_range": (15, 55),
        "volume_range": (1e5, 2e5)
    },
    {
        "name": "wider_range",
        "initial_capital": 100000,
        "range_width_percent": 0.4,
        "rebalance_threshold": 0.1,
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 4, 1),
        "eth_start_price": 2000,
        "drift": 0,
        "volatility": 0.0075,
        "random_seed": random_seed,
        "gas_price_range": (15, 55),
        "volume_range": (1e5, 2e5)
    },
    {
        "name": "higher_rebalance_threshold",
        "initial_capital": 100000,
        "range_width_percent": 0.2,
        "rebalance_threshold": 0.2,
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 4, 1),
        "eth_start_price": 2000,
        "drift": 0,
        "volatility": 0.0075,
        "random_seed": random_seed,
        "gas_price_range": (15, 55),
        "volume_range": (1e5, 2e5)
    },
    {
        "name": "high_gas_prices",
        "initial_capital": 100000,
        "range_width_percent": 0.2,
        "rebalance_threshold": 0.1,
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 4, 1),
        "eth_start_price": 2000,
        "drift": 0,
        "volatility": 0.0075,
        "random_seed": random_seed,
        "gas_price_range": (40, 120),
        "volume_range": (1e5, 2e5)
    },
    {
        "name": "high_volume",
        "initial_capital": 100000,
        "range_width_percent": 0.2,
        "rebalance_threshold": 0.1,
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 4, 1),
        "eth_start_price": 2000,
        "drift": 0,
        "volatility": 0.0075,
        "random_seed": random_seed,
        "gas_price_range": (15, 55),
        "volume_range": (5e5, 1e6)
    }
]

# Function to modify the UniswapV3Backtest class's load_market_data method to use our parameters
def modify_load_market_data(backtester, params):
    original_load_market_data = backtester.load_market_data
    
    def new_load_market_data(start_date, end_date):
        """Generate synthetic market data for the backtest period with our custom parameters"""
        dates = pd.date_range(start=start_date, end=end_date, freq='h')
        n = len(dates)
        
        # Use a single random seed
        np.random.seed(params["random_seed"])
        
        # Generate ETH price with our custom drift and volatility
        drift = params["drift"]
        volatility = params["volatility"]
        returns = np.random.normal(drift, volatility, n)
        log_prices = np.cumsum(returns) + np.log(params["eth_start_price"])
        prices = np.exp(log_prices)
        
        # Generate other market data with our custom parameters
        gas_price_min, gas_price_max = params["gas_price_range"]
        volume_min, volume_max = params["volume_range"]
        
        liquidity = np.random.uniform(1e9, 2e9, n)  # Pool liquidity
        volume = prices * np.random.uniform(volume_min, volume_max, n)  # Volume proportional to price
        gas_price = np.random.uniform(gas_price_min, gas_price_max, n)  # Gas price in Gwei
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'liquidity': liquidity,
            'volume_24h': volume,
            'gas_price_gwei': gas_price
        })
    
    # Replace the method
    backtester.load_market_data = new_load_market_data

# Create a summary results DataFrame
summary_columns = [
    "name", "total_return", "annualized_return", "annualized_volatility", 
    "max_drawdown", "sharpe_ratio", "final_nav", "initial_capital",
    "total_gas_cost", "total_swap_cost", "total_fees_earned", "rebalance_count",
    "drift", "volatility", "range_width_percent", "rebalance_threshold"
]
summary_results = pd.DataFrame(columns=summary_columns)

# Run backtests for each parameter set
for params in parameter_sets:
    print(f"\nRunning backtest: {params['name']}")
    
    # Create backtest instance with the parameters
    backtester = UniswapV3Backtest(
        initial_capital=params["initial_capital"],
        range_width_percent=params["range_width_percent"],
        rebalance_threshold=params["rebalance_threshold"]
    )
    
    # Modify the load_market_data method to use our parameters
    modify_load_market_data(backtester, params)
    
    # Run the backtest
    results = backtester.run_backtest(params["start_date"], params["end_date"])
    
    # Save NAV history to CSV
    output_dir = f"results/{params['name']}"
    os.makedirs(output_dir, exist_ok=True)
    results["nav_df"].to_csv(f"{output_dir}/nav_history.csv", index=False)
    
    # Save results data as JSON
    results_json = {
        "name": params["name"],
        "parameters": {k: str(v) if isinstance(v, datetime) else v for k, v in params.items()},
        "metrics": {
            "total_return": results["total_return"],
            "annualized_return": results["annualized_return"],
            "annualized_volatility": results["annualized_volatility"],
            "max_drawdown": results["max_drawdown"],
            "sharpe_ratio": results["sharpe_ratio"],
            "final_nav": results["final_nav"],
            "initial_capital": results["initial_capital"],
            "total_gas_cost": results["total_gas_cost"],
            "total_swap_cost": results["total_swap_cost"],
            "total_fees_earned": results["total_fees_earned"],
            "rebalance_count": results["rebalance_count"]
        }
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results_json, f, indent=4)
    
    # Create and save plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # NAV over time
    ax1.plot(results["nav_df"]["timestamp"], results["nav_df"]["nav"], label='Portfolio NAV', linewidth=2)
    ax1.axhline(y=params["initial_capital"], color='r', linestyle='--', label='Initial Capital')
    ax1.set_title('Portfolio NAV Over Time')
    ax1.set_ylabel('USDC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ETH price
    ax2.plot(results["nav_df"]["timestamp"], results["nav_df"]["price"], label='ETH Price', color='orange', linewidth=2)
    ax2.set_title('ETH/USDC Price')
    ax2.set_ylabel('USDC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cumulative return
    ax3.plot(results["nav_df"]["timestamp"], results["nav_df"]["cumulative_return"] * 100, label='Strategy Return', linewidth=2)
    ax3.set_title('Cumulative Return (%)')
    ax3.set_ylabel('Return (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Drawdown
    rolling_max = results["nav_df"]["nav"].cummax()
    drawdown = (results["nav_df"]["nav"] - rolling_max) / rolling_max * 100
    ax4.fill_between(results["nav_df"]["timestamp"], drawdown, 0, alpha=0.3, color='red')
    ax4.plot(results["nav_df"]["timestamp"], drawdown, color='red', linewidth=1)
    ax4.set_title('Drawdown (%)')
    ax4.set_ylabel('Drawdown (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fname=f"{output_dir}/backtest_results.png", dpi=300)
    plt.close()
    
    # Print summary results
    print("\n=== BACKTEST RESULTS ===")
    print(f"Name: {params['name']}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Annualized Volatility: {results['annualized_volatility']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Final NAV: ${results['final_nav']:.2f}")
    print("\n=== COSTS ===")
    print(f"Total Gas Costs: ${results['total_gas_cost']:.2f}")
    print(f"Total Swap Costs: ${results['total_swap_cost']:.2f}")
    print(f"Total Fees Earned: ${results['total_fees_earned']:.2f}")
    print(f"Number of Rebalances: {results['rebalance_count']}")
    
    # Add to summary results
    summary_results = pd.concat([summary_results, pd.DataFrame([{
        "name": params["name"],
        "total_return": results["total_return"],
        "annualized_return": results["annualized_return"],
        "annualized_volatility": results["annualized_volatility"],
        "max_drawdown": results["max_drawdown"],
        "sharpe_ratio": results["sharpe_ratio"],
        "final_nav": results["final_nav"],
        "initial_capital": results["initial_capital"],
        "total_gas_cost": results["total_gas_cost"],
        "total_swap_cost": results["total_swap_cost"],
        "total_fees_earned": results["total_fees_earned"],
        "rebalance_count": results["rebalance_count"],
        "drift": params["drift"],
        "volatility": params["volatility"],
        "range_width_percent": params["range_width_percent"],
        "rebalance_threshold": params["rebalance_threshold"]
    }])], ignore_index=True)

# Save summary results to CSV
summary_results.to_csv("results/summary_results.csv", index=False)

# Create comparison plots
plt.figure(figsize=(15, 10))

# Plot cumulative returns for all parameter sets
for params in parameter_sets:
    nav_df = pd.read_csv(f"results/{params['name']}/nav_history.csv")
    plt.plot(pd.to_datetime(nav_df["timestamp"]), nav_df["cumulative_return"] * 100, label=params["name"])

plt.title("Cumulative Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/cumulative_returns_comparison.png", dpi=300)
plt.close()

# Create a correlation matrix between NAV and ETH price for each parameter set
correlations = []
for params in parameter_sets:
    nav_df = pd.read_csv(f"results/{params['name']}/nav_history.csv")
    correlation = nav_df["nav"].corr(nav_df["price"])
    correlations.append({"name": params["name"], "correlation": correlation})

correlation_df = pd.DataFrame(correlations)
correlation_df.to_csv("results/nav_eth_correlations.csv", index=False)

# Print completion message
print("\nBacktest runs completed. Results saved to 'results/' directory.")
print(f"Summary results available in 'results/summary_results.csv'")
print(f"Correlation analysis available in 'results/nav_eth_correlations.csv'")
