import importlib
import importlib.metadata
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException

from market_data import resolve_pair_lightweight
from pydantic import BaseModel, Field


API_VERSION = "0.2.0"
CACHE_PATH = Path("/tmp/tradingstrategy-cache")
STRATEGY_FILE = Path("/app/native_strategy.py")

api = FastAPI(title="DeFi Simulator", version=API_VERSION)


def _version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _module_status(module_name: str, *dist_names: str) -> dict:
    try:
        module = importlib.import_module(module_name)
        return {
            "importable": True,
            "version": _version(*dist_names) or getattr(module, "__version__", None),
            "path": getattr(module, "__file__", None),
        }
    except Exception as exc:
        return {
            "importable": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _engine_status() -> dict:
    return {
        "tradeexecutor": _module_status("tradeexecutor", "trade-executor"),
        "eth_defi": _module_status("eth_defi", "web3-ethereum-defi"),
        "tradingstrategy": _module_status("tradingstrategy", "trading-strategy"),
        "demeter": _module_status("demeter", "zelos-demeter", "demeter"),
    }


def _api_key_is_configured() -> bool:
    return bool(os.environ.get("TRADING_STRATEGY_API_KEY"))


def _get_ts_client():
    if not _api_key_is_configured():
        raise HTTPException(status_code=503, detail="Trading Strategy API key is not configured")

    try:
        from tradingstrategy.client import Client
        return Client.create_live_client(
            api_key=os.environ["TRADING_STRATEGY_API_KEY"],
            cache_path=CACHE_PATH,
            settings_path=None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Trading Strategy client initialisation failed: {type(exc).__name__}: {exc}",
        ) from exc


def _parse_dt(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {value}") from exc
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _time_bucket(value: str):
    from tradingstrategy.timebucket import TimeBucket

    aliases = {
        "1m": TimeBucket.m1,
        "5m": TimeBucket.m5,
        "15m": TimeBucket.m15,
        "1h": TimeBucket.h1,
        "4h": TimeBucket.h4,
        "1d": TimeBucket.d1,
    }
    if value not in aliases:
        raise HTTPException(status_code=400, detail=f"Unsupported time bucket: {value}")
    return aliases[value]


def _chain_id(value: str):
    from tradingstrategy.chain import ChainId

    aliases = {
        "ethereum": ChainId.ethereum,
        "polygon": ChainId.polygon,
        "arbitrum": ChainId.arbitrum,
        "base": ChainId.base,
    }
    if value.lower() not in aliases:
        raise HTTPException(status_code=400, detail=f"Unsupported chain: {value}")
    return aliases[value.lower()]


class NativeBacktestRequest(BaseModel):
    chain: str = "ethereum"
    exchange: str = "uniswap-v3"
    base: str = "WETH"
    quote: str = "USDC"
    fee_tier: float = Field(default=0.0005, gt=0)
    start: str = "2024-01-01T00:00:00Z"
    end: str = "2024-01-08T00:00:00Z"
    time_bucket: Literal["1h"] = "1h"
    initial_capital: float = Field(default=10_000, gt=0)
    strategy: Literal["buy_and_hold"] = "buy_and_hold"
    position_size: float = Field(default=0.99, gt=0, le=1)


@api.get("/health")
def health():
    engines = _engine_status()
    imports_ok = all(item["importable"] for item in engines.values())
    return {
        "status": "ok" if imports_ok and _api_key_is_configured() else "degraded",
        "api_version": API_VERSION,
        "python": sys.version.split()[0],
        "engines": engines,
        "integration": {
            "all_engines_importable": imports_ok,
            "trading_strategy_auth": "configured" if _api_key_is_configured() else "missing",
            "native_strategy_file": STRATEGY_FILE.exists(),
        },
    }


@api.get("/capabilities")
def capabilities():
    return {
        "market_data": {
            "trading_strategy": True,
            "ohlcv": True,
            "clmm": True,
            "lending": True,
        },
        "backtesting": {
            "trade_executor_native": True,
            "strategies": ["buy_and_hold"],
            "no_lookahead": True,
        },
        "defi_simulation": {
            "demeter_adapter_importable": _engine_status()["demeter"]["importable"],
            "uniswap_v3": "clmm_adapter",
            "aave_v3": "lending_smoke_test",
        },
    }


@api.get("/selftest/trading-strategy")
@api.get("/selftest/trading_strategy")
def trading_strategy_selftest():
    try:
        from tradingstrategy.chain import ChainId
        from tradingstrategy.timebucket import TimeBucket

        client = _get_ts_client()
        pair = resolve_pair_lightweight(
            client,
            chain_id=ChainId.ethereum,
            exchange_slug="uniswap-v3",
            base_token="WETH",
            quote_token="USDC",
            fee_tier=0.0005,
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        candles = client.fetch_candles_by_pair_ids(
            [pair.pair_id],
            TimeBucket.h1,
            start_time=start,
            end_time=end,
            progress_bar_description="DeFi simulator self-test",
        )

        safe_cols = [
            col for col in ("timestamp", "open", "high", "low", "close", "volume")
            if col in candles.columns
        ]
        sample = candles[safe_cols].head(3).copy()
        if "timestamp" in sample.columns:
            sample["timestamp"] = sample["timestamp"].astype(str)

        return {
            "status": "ok",
            "authenticated": True,
            "pair": {
                "pair_id": int(pair.pair_id),
                "chain": "ethereum",
                "exchange": pair.exchange_slug,
                "base": pair.base_token_symbol,
                "quote": pair.quote_token_symbol,
                "fee_tier": float(pair.fee_tier),
            },
            "bucket": "1h",
            "requested_range": {"start": start.isoformat(), "end": end.isoformat()},
            "rows": int(len(candles)),
            "sample": sample.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Trading Strategy self-test failed: {type(exc).__name__}: {exc}",
        ) from exc


@api.post("/backtest/native")
def native_backtest(req: NativeBacktestRequest):
    if not STRATEGY_FILE.exists():
        raise HTTPException(status_code=500, detail="Native strategy file is missing")
    if not _api_key_is_configured():
        raise HTTPException(status_code=503, detail="Trading Strategy API key is not configured")

    start = _parse_dt(req.start)
    end = _parse_dt(req.end)
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be after start")
    if (end - start).days > 31:
        raise HTTPException(status_code=400, detail="Initial native endpoint is limited to 31 days per run")

    try:
        from tradeexecutor.backtest.backtest_module import run_backtest_for_module
        from tradeexecutor.cli.log import setup_logging
        from tradeexecutor.strategy.default_routing_options import TradeRouting
        from tradeexecutor.strategy.reserve_currency import ReserveCurrency
        import logging as _logging

        # run_backtest_for_module() assumes the trade-executor CLI has already
        # installed its custom Logger.trade()/trade_high() methods.
        if not hasattr(_logging.Logger, "trade"):
            setup_logging(log_level=_logging.INFO)

        chain_id = _chain_id(req.chain)
        bucket = _time_bucket(req.time_bucket)

        if (
            req.chain.lower() != "ethereum"
            or req.exchange != "uniswap-v3"
            or req.base.upper() != "WETH"
            or req.quote.upper() != "USDC"
            or abs(req.fee_tier - 0.0005) > 1e-12
        ):
            raise HTTPException(
                status_code=400,
                detail="Initial native endpoint supports only Ethereum / uniswap-v3 / WETH-USDC / 0.0005",
            )

        route = TradeRouting.uniswap_v3_usdc

        result = run_backtest_for_module(
            strategy_file=STRATEGY_FILE,
            cache_path=CACHE_PATH,
            trading_strategy_api_key=None,
            verbose=False,
            max_workers=1,
            mod_overrides={
                "CHAIN_ID": chain_id,
                "EXCHANGE_SLUG": req.exchange,
                "BASE_TOKEN": req.base.upper(),
                "QUOTE_TOKEN": req.quote.upper(),
                "FEE_TIER": req.fee_tier,
                "CANDLE_TIME_BUCKET": bucket,
                "trading_strategy_cycle": {
                    "1h": importlib.import_module("tradeexecutor.strategy.cycle").CycleDuration.cycle_1h,
                }[req.time_bucket],
                "trade_routing": route,
                "reserve_currency": ReserveCurrency.usdc,
                "backtest_start": start,
                "backtest_end": end,
                "initial_cash": req.initial_capital,
                "POSITION_SIZE": req.position_size,
            },
        )

        state = result.state
        universe = result.strategy_universe
        portfolio = state.portfolio
        final_value = float(portfolio.get_net_asset_value())
        trades = list(portfolio.get_all_trades())

        candle_range = universe.data_universe.candles.get_timestamp_range()
        pair = universe.data_universe.pairs.get_single()

        return {
            "status": "ok",
            "engine": "trade-executor",
            "data_source": "Trading Strategy",
            "strategy": req.strategy,
            "no_lookahead": True,
            "request": req.model_dump(),
            "resolved_pair": {
                "pair_id": int(pair.pair_id),
                "exchange": pair.exchange_slug,
                "base": pair.base_token_symbol,
                "quote": pair.quote_token_symbol,
                "fee_tier": float(pair.fee_tier),
            },
            "dataset": {
                "candle_range": [str(candle_range[0]), str(candle_range[1])],
                "time_bucket": req.time_bucket,
            },
            "result": {
                "initial_capital": req.initial_capital,
                "final_value": final_value,
                "total_return_pct": (final_value / req.initial_capital - 1.0) * 100.0,
                "trade_count": len(trades),
                "open_positions": len(portfolio.open_positions),
                "closed_positions": len(portfolio.closed_positions),
                "cash": float(portfolio.get_cash()),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Native trade-executor backtest failed: {type(exc).__name__}: {exc}",
        ) from exc


@api.get("/selftest/demeter")
def demeter_selftest():
    """Load real Trading Strategy CLMM data through the official Demeter adapter."""
    try:
        from demeter import MarketInfo
        from demeter.uniswap import UniLpMarket
        from tradingstrategy.chain import ChainId
        from tradingstrategy.timebucket import TimeBucket
        from tradeexecutor.strategy.demeter.adapter import (
            load_clmm_data_to_uni_lp_market,
            to_demeter_uniswap_v3_pool,
        )

        client = _get_ts_client()
        pair = resolve_pair_lightweight(
            client,
            chain_id=ChainId.ethereum,
            exchange_slug="uniswap-v3",
            base_token="WETH",
            quote_token="USDC",
            fee_tier=0.0005,
        )

        start = datetime(2024, 1, 1)
        fetch_end = datetime(2024, 1, 2)
        clmm = client.fetch_clmm_liquidity_provision_candles_by_pair_ids(
            [pair.pair_id],
            TimeBucket.m1,
            start_time=start,
            end_time=fetch_end,
            progress_bar_description="DeFi simulator Demeter self-test",
        )
        if clmm is None or len(clmm) == 0:
            raise RuntimeError("Trading Strategy returned no CLMM rows")

        pool = to_demeter_uniswap_v3_pool(pair)
        market = UniLpMarket(MarketInfo("weth_usdc_5bps"), pool)

        # The adapter expects a full calendar day and fills missing minute rows.
        load_clmm_data_to_uni_lp_market(
            market,
            clmm.copy(),
            start_date=start,
            end_date=start,
        )

        if not isinstance(market.data, pd.DataFrame) or len(market.data) == 0:
            raise RuntimeError("Demeter market data was not populated")

        price_data = market.get_price_from_data()
        if price_data is None or len(price_data) == 0:
            raise RuntimeError("Demeter could not derive prices from CLMM data")

        return {
            "status": "ok",
            "source": "Trading Strategy CLMM",
            "adapter": "tradeexecutor.strategy.demeter.adapter",
            "pair": {
                "pair_id": int(pair.pair_id),
                "chain": "ethereum",
                "exchange": pair.exchange_slug,
                "base": pair.base_token_symbol,
                "quote": pair.quote_token_symbol,
                "fee_tier": float(pair.fee_tier),
            },
            "request": {
                "bucket": "1m",
                "start": start.isoformat(),
                "end": fetch_end.isoformat(),
            },
            "clmm_rows": int(len(clmm)),
            "clmm_columns": [str(col) for col in clmm.columns],
            "demeter": {
                "market_type": type(market).__name__,
                "pool_type": type(pool).__name__,
                "tick_spacing": int(pool.tick_spacing),
                "loaded_rows": int(len(market.data)),
                "data_start": str(market.data.index.min()),
                "data_end": str(market.data.index.max()),
                "price_rows": int(len(price_data)),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Demeter CLMM self-test failed: {type(exc).__name__}: {exc}",
        ) from exc


@api.get("/selftest/aave")
def aave_selftest():
    """Run a short Aave V3 Demeter simulation using Trading Strategy lending rates."""
    try:
        from decimal import Decimal
        from pathlib import Path as _Path

        from demeter import Actuator, AtTimeTrigger, MarketInfo, MarketTypeEnum, Strategy, TokenInfo
        from demeter.aave import AaveV3Market
        from tradingstrategy.chain import ChainId
        from tradingstrategy.lending import LendingCandleType, LendingProtocolType
        from tradingstrategy.timebucket import TimeBucket

        client = _get_ts_client()
        start = pd.Timestamp("2024-01-01 00:00:00")
        end = pd.Timestamp("2024-01-02 00:00:00")

        reserve_universe = client.fetch_lending_reserve_universe()
        weth_desc = (ChainId.ethereum, LendingProtocolType.aave_v3, "WETH")
        usdc_desc = (ChainId.ethereum, LendingProtocolType.aave_v3, "USDC")
        limited = reserve_universe.limit([weth_desc, usdc_desc])
        weth_reserve = limited.resolve_lending_reserve(weth_desc)
        usdc_reserve = limited.resolve_lending_reserve(usdc_desc)

        rate_map = client.fetch_lending_candles_for_universe(
            limited,
            TimeBucket.h1,
            start_time=start,
            end_time=end,
        )

        def _rate_frame(candle_type, reserve_id):
            df = rate_map[candle_type].copy()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp", drop=False)
            df = df[df["reserve_id"] == reserve_id].sort_index()
            if len(df) == 0:
                raise RuntimeError(f"No {candle_type.value} data for reserve {reserve_id}")
            return df

        grid = pd.date_range(start=start, end=end, freq="1h")

        def _build_demeter_token_data(reserve):
            supply = _rate_frame(LendingCandleType.supply_apr, reserve.reserve_id)
            borrow = _rate_frame(LendingCandleType.variable_borrow_apr, reserve.reserve_id)

            supply_close = supply["close"].astype(float).reindex(grid).ffill().bfill()
            borrow_close = borrow["close"].astype(float).reindex(grid).ffill().bfill()

            # Trading Strategy stores APR as percent units; Demeter expects decimal annual rates.
            liquidity_rate = supply_close / 100.0
            variable_rate = borrow_close / 100.0

            seconds_per_step = 3600.0
            seconds_per_year = 31_536_000.0

            liquidity_index = []
            variable_borrow_index = []
            li = 1.0
            bi = 1.0
            for s_rate, b_rate in zip(liquidity_rate, variable_rate):
                liquidity_index.append(li)
                variable_borrow_index.append(bi)
                li *= 1.0 + float(s_rate) * seconds_per_step / seconds_per_year
                bi *= 1.0 + float(b_rate) * seconds_per_step / seconds_per_year

            return pd.DataFrame(
                {
                    "liquidity_rate": liquidity_rate.values,
                    "stable_borrow_rate": [0.0] * len(grid),
                    "variable_borrow_rate": variable_rate.values,
                    "liquidity_index": liquidity_index,
                    "variable_borrow_index": variable_borrow_index,
                },
                index=grid,
            )

        weth = TokenInfo(
            name="WETH",
            decimal=int(weth_reserve.asset_decimals),
            address=weth_reserve.asset_address,
        )
        usdc = TokenInfo(
            name="USDC",
            decimal=int(usdc_reserve.asset_decimals),
            address=usdc_reserve.asset_address,
        )

        risk_path = _Path("/tmp/aave-risk-ethereum-selftest.csv")
        risk_rows = []
        for token, reserve, fallback_ltv, fallback_lt in (
            ("WETH", weth_reserve, 0.80, 0.825),
            ("USDC", usdc_reserve, 0.75, 0.78),
        ):
            details = reserve.additional_details
            ltv = float(details.ltv) if details and details.ltv is not None else fallback_ltv
            liq_threshold = (
                float(details.liquidation_threshold)
                if details and details.liquidation_threshold is not None
                else fallback_lt
            )
            risk_rows.append(
                {
                    "symbol": token,
                    "canCollateral": True,
                    "LTV": f"{ltv * 100:.8f}%",
                    "liqThereshold": f"{liq_threshold * 100:.8f}%",
                    "liqBonus": "5%",
                    "reserveFactor": 0.1,
                    "canBorrow": True,
                    "optimalUtilization": 0.8,
                    "canBorrowStable": False,
                    "debtCeiling": 0,
                    "supplyCap": 0,
                    "borrowCap": 0,
                    "eModeLtv": 0,
                    "eModeLiquidationThereshold": 0,
                    "eModeLiquidationBonus": 0,
                    "borrowableInIsolation": False,
                }
            )
        pd.DataFrame(risk_rows).to_csv(risk_path, sep=";", index=False)

        market_key = MarketInfo("aave", MarketTypeEnum.aave_v3)
        market = AaveV3Market(
            market_info=market_key,
            risk_parameters_path=str(risk_path),
            tokens=[weth, usdc],
        )
        market.set_token_data(weth, _build_demeter_token_data(weth_reserve))
        market.set_token_data(usdc, _build_demeter_token_data(usdc_reserve))

        pair = resolve_pair_lightweight(
            client,
            chain_id=ChainId.ethereum,
            exchange_slug="uniswap-v3",
            base_token="WETH",
            quote_token="USDC",
            fee_tier=0.0005,
        )
        price_raw = client.fetch_candles_by_pair_ids(
            [pair.pair_id],
            TimeBucket.h1,
            start_time=start,
            end_time=end,
            progress_bar_description="Aave self-test WETH price",
        )
        price_raw = price_raw.copy()
        if "timestamp" in price_raw.columns:
            price_raw["timestamp"] = pd.to_datetime(price_raw["timestamp"])
            price_raw = price_raw.set_index("timestamp")
        weth_price = price_raw["close"].astype(float).reindex(grid).ffill().bfill()
        price_df = pd.DataFrame({"WETH": weth_price, "USDC": 1.0}, index=grid)

        supply_at = grid[1].to_pydatetime()
        borrow_at = grid[2].to_pydatetime()
        repay_at = grid[-2].to_pydatetime()
        withdraw_at = grid[-1].to_pydatetime()

        class _AaveSmokeStrategy(Strategy):
            def initialize(self):
                self.triggers.extend(
                    [
                        AtTimeTrigger(time=supply_at, do=self._supply),
                        AtTimeTrigger(time=borrow_at, do=self._borrow),
                        AtTimeTrigger(time=repay_at, do=self._repay),
                        AtTimeTrigger(time=withdraw_at, do=self._withdraw),
                    ]
                )

            def _supply(self, row_data):
                market.supply(weth, 1.0, True)

            def _borrow(self, row_data):
                market.borrow(usdc, 500.0)

            def _repay(self, row_data):
                for key in list(market.borrow_keys):
                    market.repay(key)

            def _withdraw(self, row_data):
                for key in list(market.supply_keys):
                    market.withdraw(key)

        actuator = Actuator()
        actuator.broker.add_market(market)
        actuator.broker.set_balance(weth, Decimal("2"))
        actuator.broker.set_balance(usdc, Decimal("10"))
        actuator.strategy = _AaveSmokeStrategy()
        actuator.set_price(price_df)
        actuator.interval = "1h"
        actuator.run(print_result=False)

        action_types = [type(action).__name__ for action in actuator.actions]
        required_actions = {"SupplyAction", "BorrowAction", "RepayAction", "WithdrawAction"}
        if not required_actions.issubset(set(action_types)):
            raise RuntimeError(f"Missing Aave actions: expected {required_actions}, got {action_types}")

        return {
            "status": "ok",
            "engine": "Demeter AaveV3Market",
            "protocol": "Aave V3",
            "chain": "ethereum",
            "period": {"start": str(start), "end": str(end), "bucket": "1h"},
            "reserves": {
                "WETH": {
                    "reserve_id": int(weth_reserve.reserve_id),
                    "ltv": float(weth_reserve.additional_details.ltv),
                    "liquidation_threshold": float(weth_reserve.additional_details.liquidation_threshold),
                },
                "USDC": {
                    "reserve_id": int(usdc_reserve.reserve_id),
                    "ltv": float(usdc_reserve.additional_details.ltv),
                    "liquidation_threshold": float(usdc_reserve.additional_details.liquidation_threshold),
                },
            },
            "actions": action_types,
            "action_count": len(action_types),
            "data_lineage": {
                "observed_historical": [
                    "Trading Strategy Aave V3 supply APR",
                    "Trading Strategy Aave V3 variable borrow APR",
                    "Trading Strategy WETH/USDC OHLCV price",
                ],
                "derived": [
                    "liquidity_index from observed supply APR",
                    "variable_borrow_index from observed variable borrow APR",
                ],
                "parameterized_or_snapshot": [
                    "stable_borrow_rate fixed to 0 because smoke uses variable debt only",
                    "non-LTV risk columns required by Demeter use smoke-test parameters",
                    "LTV/liquidation threshold use Trading Strategy reserve metadata snapshot frozen for this test",
                ],
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Aave V3 self-test failed: {type(exc).__name__}: {exc}",
        ) from exc


@api.get("/selftest/native-backtest")
@api.get("/selftest/native_backtest")
def native_backtest_selftest():
    """Run the default native trade-executor + Trading Strategy backtest smoke test."""
    return native_backtest(NativeBacktestRequest())
