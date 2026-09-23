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
            "uniswap_v3": "adapter_next",
            "aave_v3": "adapter_next",
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
    """Verify the official trade-executor -> Demeter adapter imports in this runtime."""
    try:
        from tradeexecutor.strategy.demeter.adapter import (
            load_clmm_data_to_uni_lp_market,
            to_demeter_token,
            to_demeter_uniswap_v3_pool,
        )
        return {
            "status": "ok",
            "adapter": "tradeexecutor.strategy.demeter.adapter",
            "functions": [
                to_demeter_token.__name__,
                to_demeter_uniswap_v3_pool.__name__,
                load_clmm_data_to_uni_lp_market.__name__,
            ],
            "clmm_execution": "not_run",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Demeter adapter import failed: {type(exc).__name__}: {exc}",
        ) from exc


@api.get("/selftest/native-backtest")
@api.get("/selftest/native_backtest")
def native_backtest_selftest():
    """Run the default native trade-executor + Trading Strategy backtest smoke test."""
    return native_backtest(NativeBacktestRequest())
