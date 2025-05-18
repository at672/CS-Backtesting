import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class Position:
    """Represents a Uniswap V3 liquidity position"""
    lower_price: float
    upper_price: float
    liquidity: float
    token0_amount: float  # ETH
    token1_amount: float  # USDC
    fees_earned0: float
    fees_earned1: float
    created_at: datetime

class UniswapV3Backtest:
    """Simplified Uniswap V3 backtester"""
    
    def __init__(self, initial_capital=100000, range_width_percent=0.2, rebalance_threshold=0.1):
        self.initial_capital = initial_capital
        self.range_width_percent = range_width_percent
        self.rebalance_threshold = rebalance_threshold
        
        self.current_position = None
        self.nav_history = []
        self.transaction_history = []
        self.total_gas_cost = 0
        self.total_swap_cost = 0
        self.total_fees_earned = 0
    
    def load_market_data(self, start_date, end_date):
        """Generate synthetic market data for the backtest period"""
        dates = pd.date_range(start=start_date, end=end_date, freq='h')
        n = len(dates)
        
        # Generate ETH price with trend and volatility
        np.random.seed(42)
        np.random.seed(420)
        # np.random.seed(5656)
        # drift = 0.0005  # Hourly drift: positive=up, negative=down, zero=sideways
        drift = 0
        volatility = 0.002  # More realistic hourly volatility (was 0.01)
        returns = np.random.normal(drift, volatility, n)
        log_prices = np.cumsum(returns) + np.log(2000)
        prices = np.exp(log_prices)
        # Generate other market data
        liquidity = np.random.uniform(1e9, 2e9, n)  # Pool liquidity
        volume = prices * np.random.uniform(1e5, 2e5, n)  # Volume proportional to price
        gas_price = np.random.uniform(15, 55, n)  # Gas price in Gwei
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'liquidity': liquidity,
            'volume_24h': volume,
            'gas_price_gwei': gas_price
        })
    
    def calculate_liquidity(self, price: float, lower_price: float, upper_price: float,
                            amount0: float, amount1: float) -> float:
        # precompute sqrt values once
        sqrt_p      = math.sqrt(price)
        sqrt_lower  = math.sqrt(lower_price)
        sqrt_upper  = math.sqrt(upper_price)

        # entirely in token0 (price below range)
        if price <= lower_price:
            return amount0 * (sqrt_lower * sqrt_upper) / (sqrt_upper - sqrt_lower)

        # entirely in token1 (price above range)
        if price >= upper_price:
            return amount1 / (sqrt_upper - sqrt_lower)

        # in-range: take the minimum of the two possible Ls
        liquidity0 = amount0 * (sqrt_upper * sqrt_p) / (sqrt_upper - sqrt_p)
        liquidity1 = amount1 / (sqrt_p - sqrt_lower)
        return min(liquidity0, liquidity1)

    def calculate_amounts(self,
                        liquidity: float,
                        price: float,
                        lower_price: float,
                        upper_price: float
                        ) -> Tuple[float, float]:
        """
        Given a Uniswap V3 liquidity L and price band [lower_price, upper_price],
        return the amounts of token0 and token1 represented by L at the current price.
        """
        sqrt_p      = math.sqrt(price)
        sqrt_lower  = math.sqrt(lower_price)
        sqrt_upper  = math.sqrt(upper_price)
        # Price at or below the band → all in token0
        if price <= lower_price:
            amount0 = liquidity * (sqrt_upper - sqrt_lower) / (sqrt_lower * sqrt_upper)
            amount1 = 0.0
        # Price at or above the band → all in token1
        elif price >= upper_price:
            amount0 = 0.0
            amount1 = liquidity * (sqrt_upper - sqrt_lower)
        # Price inside the band → split according to distance from each tick
        else:
            amount0 = liquidity * (sqrt_upper - sqrt_p) / (sqrt_p * sqrt_upper)
            amount1 = liquidity * (sqrt_p - sqrt_lower)
        return amount0, amount1

    def should_rebalance(self, position, current_price):
        """Determine if position should be rebalanced"""
        if current_price <= position.lower_price or current_price >= position.upper_price:
            return True
        
        range_size = position.upper_price - position.lower_price
        distance_to_lower = current_price - position.lower_price
        distance_to_upper = position.upper_price - current_price
        
        return (distance_to_lower / range_size < self.rebalance_threshold or
                distance_to_upper / range_size < self.rebalance_threshold)
    
    def calculate_price_range(self, current_price):
        """Calculate new price range centered around current price"""
        half_width = self.range_width_percent / 2
        lower_price = current_price * (1 - half_width)
        upper_price = current_price * (1 + half_width)
        return lower_price, upper_price
    
    def calculate_gas_cost(self, gas_price_gwei, operation):
        """Calculate gas cost in USDC"""
        gas_limits = {
            'add_liquidity': 150000,
            'remove_liquidity': 120000,
            'collect_fees': 80000,
            'rebalance': 250000
        }
        #Convert from giga wei to base
        gas_price_eth = gas_price_gwei * 1e9 / 1e18
        gas_used = gas_limits.get(operation, 200000)
        
        return gas_price_eth * gas_used
    
    def calculate_swap_impact(self, amount, liquidity):
        """Calculate swap price impact and fees"""
        impact = (amount / liquidity) * 0.5
        fee = 0.0005  # 0.05% fee tier
        
        return impact + fee
    
    def calculate_optimal_amounts(self, total_value, current_price, lower_price, upper_price):
        """Calculate optimal token amounts for a given price range"""
        # Calculate geometric mean of the range
        price_geometric_mean = math.sqrt(lower_price * upper_price)
        
        # Special cases: price outside range
        if current_price <= lower_price:
            # All in token0 (ETH)
            return total_value / current_price, 0
        elif current_price >= upper_price:
            # All in token1 (USDC)
            return 0, total_value
        
        # Calculate the optimal ratio based on current price relative to range
        sqrt_p = math.sqrt(current_price)
        sqrt_lower = math.sqrt(lower_price)
        sqrt_upper = math.sqrt(upper_price)
        
        # Calculate liquidity value L
        L = total_value / ((sqrt_upper - sqrt_p) / (sqrt_p * sqrt_upper) * current_price + (sqrt_p - sqrt_lower))
        
        # Calculate optimal amounts
        token0 = L * (sqrt_upper - sqrt_p) / (sqrt_p * sqrt_upper)
        token1 = L * (sqrt_p - sqrt_lower)
        
        return token0, token1

    
    def estimate_fees(self, position, price, volume_24h, pool_liquidity, hours):
        """Estimate fee earnings for a position"""
        # ONLY earns fees if I'm in the range.
        if price < position.lower_price or price > position.upper_price:
            return 0, 0
        
        position_share = position.liquidity / pool_liquidity
        fee_tier = 0.0005  # 0.05% fee tier
        volume_period = volume_24h * (hours / 24)
        fee_amount = volume_period * fee_tier * position_share
        
        token0_fees = fee_amount * 0.5 / price  # Convert half to ETH
        token1_fees = fee_amount * 0.5  # Half in USDC
        
        return token0_fees, token1_fees
    
    def run_backtest(self, start_date, end_date):
        """Run backtest simulation"""
        print(f"Running backtest from {start_date} to {end_date}")
        
        # Load market data
        market_data = self.load_market_data(start_date, end_date)
        
        # Initialize position with first market state
        first_row = market_data.iloc[0]
        initial_price = first_row['price']
        lower_price, upper_price = self.calculate_price_range(initial_price)
        
        # Split initial capital 50/50
        initial_eth = (self.initial_capital / 2) / initial_price
        initial_usdc = self.initial_capital / 2
        
        # Calculate the theoretically optimal amounts for the given range
        price_geometric_mean = math.sqrt(lower_price * upper_price)
        ratio = math.sqrt(initial_price / price_geometric_mean)


        # If price = sqrt(lower * upper), you get perfect 50/50
        # Otherwise, adjust the ratio
        optimal_eth = self.initial_capital / (initial_price + price_geometric_mean)
        optimal_usdc = optimal_eth * price_geometric_mean

        # Initialize with these optimal amounts
        initial_liquidity = self.calculate_liquidity(
            initial_price, lower_price, upper_price, optimal_eth, optimal_usdc)
        
        # Calculate correct token amounts from this liquidity
        actual_token0, actual_token1 = self.calculate_amounts(
            initial_liquidity, initial_price, lower_price, upper_price)

        # After calculating initial_liquidity and getting actual_token0, actual_token1
        position_value = actual_token0 * initial_price + actual_token1
        print(f"Initial capital: ${self.initial_capital}")
        print(f"Initial position value: ${position_value}")
        print(f"Difference: ${self.initial_capital - position_value} ({(self.initial_capital - position_value) / self.initial_capital * 100:.2f}%)")

        # Create initial position
        self.current_position = Position(
            lower_price=lower_price,
            upper_price=upper_price,
            liquidity=initial_liquidity,
            token0_amount=actual_token0,
            token1_amount=actual_token1,
            fees_earned0=0,
            fees_earned1=0,
            created_at=first_row['timestamp']
        )
        
        # Record initial gas cost
        gas_cost = self.calculate_gas_cost(first_row['gas_price_gwei'], 'add_liquidity')
        self.total_gas_cost += gas_cost * initial_price  # Convert to USDC
        
        # Calculate initial NAV correctly
        self.initial_nav = self._calculate_nav(initial_price)  # Use the same NAV calculation as later iterations

        # Track NAV
        self.nav_history.append({
            'timestamp': first_row['timestamp'],
            'nav': self.initial_nav,
            'price': initial_price
        })
        
        # Main simulation loop
        prev_time = first_row['timestamp']
        for i in range(1, len(market_data)):
            row = market_data.iloc[i]
            current_time = row['timestamp']
            current_price = row['price']
            time_diff = (current_time - prev_time).total_seconds() / 3600  # Hours
            
            price_in_range = (current_price >= self.current_position.lower_price and 
                  current_price <= self.current_position.upper_price)
            
            # Update position token amounts based on new price
            #if self.current_position and price_in_range:
            if self.current_position:
                self.current_position.token0_amount, self.current_position.token1_amount = \
                    self.calculate_amounts(self.current_position.liquidity, current_price, 
                                        self.current_position.lower_price, 
                                        self.current_position.upper_price)

            # Collect fee earnings
            if self.current_position:
                token0_fees, token1_fees = self.estimate_fees(
                    self.current_position,
                    current_price,
                    row['volume_24h'],
                    row['liquidity'],
                    time_diff
                )
                self.current_position.fees_earned0 += token0_fees
                self.current_position.fees_earned1 += token1_fees
                self.total_fees_earned += token0_fees * current_price + token1_fees
            
            # Check if rebalance is needed
            if self.should_rebalance(self.current_position, current_price):
                self._rebalance(row)
            
            # Calculate current NAV
            nav = self._calculate_nav(current_price)
            
            # Record NAV
            self.nav_history.append({
                'timestamp': current_time,
                'nav': nav,
                'price': current_price
            })
            
            prev_time = current_time
        
        # Final calculations and results
        return self._calculate_results()
    
    def _rebalance(self, market_row):
        """Rebalance the current position"""
        current_price = market_row['price']
        current_time = market_row['timestamp']
        
        # Get current position value (tokens + fees)
        token0_amount = self.current_position.token0_amount + self.current_position.fees_earned0
        token1_amount = self.current_position.token1_amount + self.current_position.fees_earned1
        
        # Total value in USDC
        total_value = token0_amount * current_price + token1_amount
        
        # Calculate gas cost
        gas_cost = self.calculate_gas_cost(market_row['gas_price_gwei'], 'rebalance')
        gas_cost_usdc = gas_cost * current_price
        self.total_gas_cost += gas_cost_usdc
        
        # Calculate swap costs (to rebalance to 50/50)
        token0_value = token0_amount * current_price # ETH balance * USDC/ETH => USDC
        token1_value = token1_amount # already in USDC. token1 is USDC
        
        target_value = (total_value - gas_cost_usdc) / 2
        swap_amount = abs(token0_value - target_value) #USDC
        print("DEBUG: swap_amount", swap_amount)
        swap_cost = self.calculate_swap_impact(swap_amount, market_row['liquidity']) * swap_amount
        self.total_swap_cost += swap_cost
        
        # Adjusted total after costs
        adjusted_total = total_value - gas_cost_usdc - swap_cost
        
        # New price range
        lower_price, upper_price = self.calculate_price_range(current_price)

        # New token amounts (optimal distribution)
        new_token0, new_token1 = self.calculate_optimal_amounts(
            adjusted_total, current_price, lower_price, upper_price)

        # Calculate new liquidity
        new_liquidity = self.calculate_liquidity(
            current_price, lower_price, upper_price, new_token0, new_token1)
        
        actual_token0, actual_token1 = self.calculate_amounts(new_liquidity, current_price, lower_price, upper_price)

        # Create new position
        self.current_position = Position(
            lower_price=lower_price,
            upper_price=upper_price,
            liquidity=new_liquidity,
            token0_amount=actual_token0,
            token1_amount=actual_token1,
            fees_earned0=0,
            fees_earned1=0,
            created_at=current_time
        )
        
        # Record transaction
        self.transaction_history.append({
            'timestamp': current_time,
            'type': 'rebalance',
            'gas_cost': gas_cost_usdc,
            'swap_cost': swap_cost,
            'price': current_price,
            'range': f"${lower_price:.2f}-${upper_price:.2f}"
        })
    
    def _calculate_nav(self, current_price):
        """Calculate current NAV"""
        # If no position, return initial capital

        if not self.current_position:
            return self.initial_capital
        
        # NAV = (current_amount1 + fee1)  
        #     + (current_amount0 + fee0) * price  
        #     – cumulative gas costs  
        #     – cumulative swap impact costs  

        # Token values
        token0_value = self.current_position.token0_amount * current_price
        token1_value = self.current_position.token1_amount
        
        # Fee values
        fee0_value = self.current_position.fees_earned0 * current_price
        fee1_value = self.current_position.fees_earned1
        
        # Total value minus costs
        nav = token0_value + token1_value + fee0_value + fee1_value
        nav -= self.total_gas_cost + self.total_swap_cost
        
        return nav
    
    def _calculate_results(self):
        """Calculate final backtest results and metrics"""
        # Convert history to DataFrame
        nav_df = pd.DataFrame(self.nav_history)
        
        # Calculate returns
        nav_df['daily_return'] = nav_df['nav'].pct_change(24)  # 24 hours = daily
        nav_df['cumulative_return'] = nav_df['nav'] / self.initial_capital - 1
        
        # Calculate metrics
        total_return = nav_df['nav'].iloc[-1] / self.initial_capital - 1
        
        # Annualized return
        days = (nav_df['timestamp'].iloc[-1] - nav_df['timestamp'].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1
        
        # Volatility
        daily_returns = nav_df['daily_return'].dropna()
        annualized_volatility = daily_returns.std() * np.sqrt(365)
        
        # Max drawdown
        rolling_max = nav_df['nav'].cummax()
        drawdown = (nav_df['nav'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (risk-free rate at 4.5%)
        risk_free_rate = 0.045
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        return {
            'nav_df': nav_df,
            'transactions': self.transaction_history,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_nav': nav_df['nav'].iloc[-1],
            'initial_capital': self.initial_capital,
            'total_gas_cost': self.total_gas_cost,
            'total_swap_cost': self.total_swap_cost,
            'total_fees_earned': self.total_fees_earned,
            'rebalance_count': len(self.transaction_history)
        }
    
    def plot_results(self, results):
        """Plot backtest results"""
        nav_df = results['nav_df']
        nav_df.to_csv('nav_history.csv', index = False)
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # NAV over time
        ax1.plot(nav_df['timestamp'], nav_df['nav'], label='Portfolio NAV', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        ax1.set_title('Portfolio NAV Over Time')
        ax1.set_ylabel('USDC')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ETH price
        ax2.plot(nav_df['timestamp'], nav_df['price'], label='ETH Price', color='orange', linewidth=2)
        ax2.set_title('ETH/USDC Price')
        ax2.set_ylabel('USDC')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cumulative return
        ax3.plot(nav_df['timestamp'], nav_df['cumulative_return'] * 100, label='Strategy Return', linewidth=2)
        ax3.set_title('Cumulative Return (%)')
        ax3.set_ylabel('Return (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = nav_df['nav'].cummax()
        drawdown = (nav_df['nav'] - rolling_max) / rolling_max * 100
        ax4.fill_between(nav_df['timestamp'], drawdown, 0, alpha=0.3, color='red')
        ax4.plot(nav_df['timestamp'], drawdown, color='red', linewidth=1)
        ax4.set_title('Drawdown (%)')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(fname = 'backtest_results.png', dpi = 600)
        
        # Print summary statistics
        print("\n=== BACKTEST RESULTS ===")
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

# Example usage
if __name__ == "__main__":
    # Create backtester with parameters
    backtester = UniswapV3Backtest(
        initial_capital=100000,  # $100,000 USDC
        range_width_percent=0.3,  # 30% wide price range
        rebalance_threshold=0.1   # Rebalance when price is within 10% of range bounds
    )
    
    # Define backtest period (at least 90 days)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 4, 1)  # 90+ days
    
    # Run backtest
    results = backtester.run_backtest(start_date, end_date)
    
    # Plot results
    backtester.plot_results(results)