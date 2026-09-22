"""Minimal native trade-executor strategy used by the API integration test.

The strategy deliberately has no custom indicator logic: it opens one WETH/USDC
spot position on the first decision cycle and then holds it. Market data and
execution are handled by trade-executor / Trading Strategy.
"""
import datetime
from typing import Dict, List

import pandas as pd

from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions

from market_data import resolve_pair_dataframe


trading_strategy_engine_version = "0.3"
trading_strategy_type = StrategyType.managed_positions
trade_routing = TradeRouting.uniswap_v3_usdc
trading_strategy_cycle = CycleDuration.cycle_1h
reserve_currency = ReserveCurrency.usdc

# These globals are intentionally overridable through run_backtest_for_module(mod_overrides=...)
CHAIN_ID = ChainId.ethereum
EXCHANGE_SLUG = "uniswap-v3"
BASE_TOKEN = "WETH"
QUOTE_TOKEN = "USDC"
FEE_TIER = 0.0005
CANDLE_TIME_BUCKET = TimeBucket.h1
backtest_start = datetime.datetime(2024, 1, 1)
backtest_end = datetime.datetime(2024, 1, 8)
initial_cash = 10_000
POSITION_SIZE = 0.99


def decide_trades(
    timestamp: pd.Timestamp,
    universe: Universe,
    state: State,
    pricing_model: PricingModel,
    cycle_debug_data: Dict,
) -> List[TradeExecution]:
    """Open once and hold. The backtest engine values the open position at the end."""
    pair = universe.pairs.get_single()
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    if position_manager.is_any_open():
        return []

    cash = state.portfolio.get_cash()
    if cash <= 0:
        return []

    amount = cash * POSITION_SIZE
    return position_manager.open_1x_long(pair, amount)


def create_trading_universe(
    ts: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    _, pair_df = resolve_pair_dataframe(
        client,
        chain_id=CHAIN_ID,
        exchange_slug=EXCHANGE_SLUG,
        base_token=BASE_TOKEN,
        quote_token=QUOTE_TOKEN,
        fee_tier=FEE_TIER,
    )

    dataset = load_partial_data(
        client,
        execution_context=execution_context,
        time_bucket=CANDLE_TIME_BUCKET,
        universe_options=universe_options,
        pairs=pair_df,
        required_history_period=datetime.timedelta(hours=2),
        reserve_asset=QUOTE_TOKEN,
        # The native backtest pricing model asks candle lookups to explicitly
        # ignore synthetic forward-filled rows. Enabling forward fill here
        # guarantees the candle frame carries the "forward_filled" marker even
        # when this liquid pool has no gaps in the requested period.
        forward_fill=True,
    )

    return TradingStrategyUniverse.create_from_dataset(dataset)
