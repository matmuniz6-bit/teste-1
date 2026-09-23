import importlib
import importlib.metadata
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
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


class AaveBacktestRequest(BaseModel):
    start: str = "2024-01-01T00:00:00Z"
    end: str = "2024-01-02T00:00:00Z"
    collateral_weth: float = Field(default=1.0, gt=0)
    borrow_usdc: float = Field(default=500.0, gt=0)
    initial_weth_balance: float = Field(default=2.0, gt=0)
    initial_usdc_balance: float = Field(default=10.0, ge=0)


class LongReversalBacktestRequest(BaseModel):
    """Study-like ETH intraday strategy adapted to our WETH/USDC market."""

    start: str = "2026-08-01T00:00:00Z"
    end: str = "2026-09-01T00:00:00Z"
    initial_capital: float = Field(default=100, gt=0)
    transaction_costs_bps: list[int] = Field(default_factory=lambda: [0, 1, 2])


class NavierStokesDayRequest(BaseModel):
    """Causal Navier-Stokes-inspired market microstructure experiment."""

    date: str = "2026-08-15"
    days: int = Field(default=1, ge=1, le=366)
    integration_gain: float = Field(default=0.10, gt=0, le=1)
    illustrative_cost_bps_per_turnover: float = Field(default=5.0, ge=0, le=100)


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
            "strategies": ["buy_and_hold", "long_reversal_study_like"],
            "no_lookahead": True,
        },
        "defi_simulation": {
            "demeter_adapter_importable": _engine_status()["demeter"]["importable"],
            "uniswap_v3": "clmm_adapter",
            "aave_v3": "backtest_endpoint",
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
            setup_logging(log_level=_logging.WARNING)

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


@api.post("/experiment/ns-criticality-day")
def ns_criticality_day(req: NavierStokesDayRequest):
    """Causal Navier-Stokes-inspired experiment on WETH/USDC.

    This is an exploratory mapping, not a claim that financial prices obey the
    physical Navier-Stokes PDE. All variables at hour t only determine the
    position applied to hour t+1.
    """
    try:
        from tradingstrategy.chain import ChainId
        from tradingstrategy.timebucket import TimeBucket

        target = pd.Timestamp(req.date)
        if target.tzinfo is not None:
            target = target.tz_convert("UTC").tz_localize(None)
        target = target.floor("D")
        target_end = target + pd.Timedelta(days=req.days)
        fetch_start = target - pd.Timedelta(hours=72)
        fetch_end = target_end + pd.Timedelta(hours=1)

        client = _get_ts_client()
        pair = resolve_pair_lightweight(
            client,
            chain_id=ChainId.ethereum,
            exchange_slug="uniswap-v3",
            base_token="WETH",
            quote_token="USDC",
            fee_tier=0.0005,
        )

        candles = client.fetch_candles_by_pair_ids(
            [pair.pair_id],
            TimeBucket.h1,
            start_time=fetch_start.to_pydatetime(),
            end_time=fetch_end.to_pydatetime(),
            progress_bar_description=f"N-S {req.days}-day OHLCV",
        )
        if candles is None or len(candles) == 0:
            raise RuntimeError("No hourly OHLCV data")

        h = candles.copy()
        if "timestamp" in h.columns:
            h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True).dt.tz_convert(None)
        elif isinstance(h.index, pd.DatetimeIndex):
            h["timestamp"] = pd.to_datetime(h.index, utc=True).tz_convert(None)
        else:
            raise RuntimeError("Hourly data has no timestamp")
        h = h.reset_index(drop=True)
        for col in ("close", "volume"):
            h[col] = pd.to_numeric(h[col], errors="coerce")
        h = (
            h[["timestamp", "close", "volume"]]
            .dropna()
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .set_index("timestamp")
        )
        if h.empty:
            raise RuntimeError("No usable hourly OHLCV data")

        # Minute CLMM history is fetched in bounded chunks so a 12-month
        # experiment does not depend on one very large API response.
        clmm_parts = []
        chunk_start = fetch_start
        while chunk_start < fetch_end:
            chunk_end = min(chunk_start + pd.Timedelta(days=31), fetch_end)
            part = client.fetch_clmm_liquidity_provision_candles_by_pair_ids(
                [pair.pair_id],
                TimeBucket.m1,
                start_time=chunk_start.to_pydatetime(),
                end_time=chunk_end.to_pydatetime(),
                progress_bar_description=f"N-S CLMM {chunk_start.date()}",
            )
            if part is not None and len(part):
                clmm_parts.append(part)
            chunk_start = chunk_end

        if not clmm_parts:
            raise RuntimeError("No CLMM liquidity data")
        q = pd.concat(clmm_parts, ignore_index=True)
        ts_col = "bucket" if "bucket" in q.columns else "timestamp"
        if ts_col not in q.columns:
            raise RuntimeError("CLMM data has no bucket/timestamp column")
        q["timestamp"] = pd.to_datetime(q[ts_col], utc=True).dt.tz_convert(None)
        q = q.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        for col in (
            "current_liquidity", "open_tick", "close_tick", "high_tick", "low_tick",
            "in_amount0", "in_amount1", "net_amount0", "net_amount1",
        ):
            if col in q.columns:
                q[col] = pd.to_numeric(q[col], errors="coerce")
        if "current_liquidity" not in q.columns:
            raise RuntimeError("CLMM data has no current_liquidity")

        agg_map = {"current_liquidity": "median"}
        if "high_tick" in q.columns and "low_tick" in q.columns:
            q["tick_span"] = (q["high_tick"] - q["low_tick"]).abs()
            agg_map["tick_span"] = "median"
        for col in ("in_amount0", "in_amount1"):
            if col in q.columns:
                agg_map[col] = "sum"
        for col in ("net_amount0", "net_amount1"):
            if col in q.columns:
                q[f"abs_{col}"] = q[col].abs()
                agg_map[f"abs_{col}"] = "sum"

        q_idx = q.set_index("timestamp").sort_index()
        liq = q_idx.resample("1h").agg(agg_map)

        # V4 spatial proxies from the minute-level price path.
        # These describe where swaps moved the price through tick space; they
        # are NOT a full LP liquidity distribution across initialized ticks.
        if "close_tick" in q_idx.columns:
            tick_stats = q_idx["close_tick"].resample("1h").agg(["std", "min", "max"])
            tick_stats = tick_stats.rename(columns={
                "std": "tick_path_dispersion",
                "min": "tick_path_min",
                "max": "tick_path_max",
            })
            tick_stats["tick_path_range"] = (
                tick_stats["tick_path_max"] - tick_stats["tick_path_min"]
            ).abs()
            liq = liq.join(tick_stats, how="left")

        if "high_tick" in q_idx.columns and "low_tick" in q_idx.columns:
            high_hour = q_idx["high_tick"].resample("1h").max()
            low_hour = q_idx["low_tick"].resample("1h").min()
            liq["tick_extreme_range"] = (high_hour - low_hour).abs()

        frame = h.join(liq, how="inner")
        frame = frame[
            (frame.index >= fetch_start)
            & (frame.index <= target_end)
        ].copy()
        if len(frame) < 48:
            raise RuntimeError(f"Insufficient joined warmup data: {len(frame)} rows")

        eps = 1e-9
        frame["log_ret"] = np.log(frame["close"]).diff()
        vol_med = frame["volume"].rolling(24, min_periods=12).median()
        frame["volume_ratio"] = (frame["volume"] / vol_med.replace(0, np.nan)).clip(0.05, 20.0)

        # u: volume-weighted hourly price velocity, then normalized to local scale.
        frame["u_raw"] = frame["log_ret"] * np.sqrt(frame["volume_ratio"])
        u_scale = frame["u_raw"].rolling(24, min_periods=12).std().replace(0, np.nan)
        frame["u"] = (frame["u_raw"] / u_scale).clip(-8, 8)

        # rho: active Uniswap V3 liquidity around the current tick, normalized
        # by its trailing 24h median. This is a density proxy, not physical mass.
        liq_med = frame["current_liquidity"].rolling(24, min_periods=12).median()
        frame["rho"] = (
            frame["current_liquidity"] / liq_med.replace(0, np.nan)
        ).clip(0.10, 10.0)

        # p: local valuation/pressure proxy = displacement from trailing 6h
        # volume-weighted price. dp is its first difference.
        pv = (frame["close"] * frame["volume"]).rolling(6, min_periods=3).sum()
        vv = frame["volume"].rolling(6, min_periods=3).sum().replace(0, np.nan)
        frame["p"] = np.log(frame["close"] / (pv / vv))
        frame["dp"] = frame["p"].diff()
        dp_scale = frame["dp"].rolling(24, min_periods=12).std().replace(0, np.nan)
        frame["dp_z"] = (frame["dp"] / dp_scale).clip(-8, 8)

        # nu: short-horizon realized volatility relative to its 24h baseline.
        rv6 = frame["log_ret"].rolling(6, min_periods=3).std()
        rv24 = frame["log_ret"].rolling(24, min_periods=12).std().replace(0, np.nan)
        frame["nu"] = (rv6 / rv24).clip(0.10, 5.0)

        frame["du"] = frame["u"].diff()
        frame["d2u"] = frame["du"].diff()

        # Dimensionless N-S-inspired components on comparable local scales.
        frame["A_advection"] = -frame["u"] * frame["du"]
        frame["P_pressure"] = -frame["dp_z"] / frame["rho"].clip(lower=0.25)
        frame["D_diffusion"] = frame["nu"] * frame["d2u"]
        frame["net_force"] = (
            frame["A_advection"] + frame["P_pressure"] + frame["D_diffusion"]
        )

        # OpenAI-inspired diagnostics: flow/load concentration, nonlinear
        # amplification vs diffusion, and hidden stress from term cancellation.
        frame["flow_to_liquidity"] = frame["u"].abs() / frame["rho"].clip(lower=0.10)
        frame["amplification_minus_damping"] = (
            (frame["A_advection"] + frame["P_pressure"]).abs()
            - frame["D_diffusion"].abs()
        )
        frame["cancellation_stress"] = (
            frame["A_advection"].abs()
            + frame["P_pressure"].abs()
            + frame["D_diffusion"].abs()
        ) / (frame["net_force"].abs() + 0.05)

        def rolling_z(s: pd.Series) -> pd.Series:
            mean = s.rolling(24, min_periods=12).mean()
            std = s.rolling(24, min_periods=12).std().replace(0, np.nan)
            return ((s - mean) / std).clip(-6, 6)

        frame["z_concentration"] = rolling_z(frame["flow_to_liquidity"])
        frame["z_amplification"] = rolling_z(frame["amplification_minus_damping"])
        frame["z_cancellation"] = rolling_z(np.log1p(frame["cancellation_stress"]))
        frame["criticality"] = (
            frame["z_concentration"]
            + frame["z_amplification"]
            + frame["z_cancellation"]
        ) / 3.0

        # V2 instability detector:
        # 1) use actual Uniswap swap-flow intensity rather than price-volume only,
        # 2) emphasize flow relative to active liquidity,
        # 3) retain nonlinear amplification vs diffusion,
        # 4) drop cancellation from the predictive score because August V1 showed
        #    essentially no next-hour magnitude relationship for that term.
        def _ratio_to_trailing_median(s: pd.Series, window: int = 72) -> pd.Series:
            med = s.rolling(window, min_periods=24).median().shift(1)
            return (s / med.replace(0, np.nan)).clip(0.0, 25.0)

        flow_components = []
        for col in ("in_amount0", "in_amount1", "abs_net_amount0", "abs_net_amount1"):
            if col in frame.columns:
                flow_components.append(_ratio_to_trailing_median(frame[col].abs()))
        if flow_components:
            flow_df = pd.concat(flow_components, axis=1)
            frame["swap_flow_intensity"] = flow_df.mean(axis=1, skipna=True)
            frame["swap_flow_intensity"] = (
                frame["swap_flow_intensity"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(frame["volume_ratio"])
            )
        else:
            # Fallback remains causal if a historical CLMM schema lacks flow columns.
            frame["swap_flow_intensity"] = frame["volume_ratio"]

        frame["flow_pressure_v2"] = (
            frame["swap_flow_intensity"] / frame["rho"].clip(lower=0.10)
        )
        frame["liquidity_deficit_v2"] = 1.0 / frame["rho"].clip(lower=0.10)
        frame["amplification_v2"] = frame["amplification_minus_damping"].clip(lower=0.0)

        def causal_z(s: pd.Series, window: int = 72) -> pd.Series:
            mean = s.rolling(window, min_periods=24).mean().shift(1)
            std = s.rolling(window, min_periods=24).std().shift(1)
            z = (s - mean) / std.replace(0, np.nan)
            # No historical variance means "no standardized anomaly", not missing data.
            return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6, 6)

        frame["z_flow_pressure_v2"] = causal_z(np.log1p(frame["flow_pressure_v2"]))
        frame["z_amplification_v2"] = causal_z(np.log1p(frame["amplification_v2"]))
        frame["z_liquidity_deficit_v2"] = causal_z(np.log1p(frame["liquidity_deficit_v2"]))
        frame["instability_v2"] = (
            frame["z_flow_pressure_v2"]
            + frame["z_amplification_v2"]
            + frame["z_liquidity_deficit_v2"]
        ) / 3.0

        # Adaptive stress threshold: top decile relative to the PREVIOUS 7 days.
        # shift(1) prevents the current observation from setting its own threshold.
        frame["instability_v2_threshold"] = (
            frame["instability_v2"]
            .rolling(168, min_periods=48)
            .quantile(0.90)
            .shift(1)
        )
        frame["high_instability_v2"] = (
            frame["instability_v2"] > frame["instability_v2_threshold"]
        )

        # V4: spatial-localisation approximation inspired by the newer
        # Navier-Stokes concentration mechanism.
        # We only have the realised minute-by-minute tick path, not the full
        # Uniswap LP distribution across initialized ticks.
        spatial_width = (
            frame["tick_path_dispersion"]
            if "tick_path_dispersion" in frame.columns
            else frame.get("tick_span", pd.Series(index=frame.index, dtype=float))
        )
        width_med = spatial_width.rolling(72, min_periods=24).median().shift(1)
        frame["spatial_width_ratio_v4"] = (
            spatial_width / width_med.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.05, 20.0)

        # Concentrated throughput: large swap flow through a locally narrow
        # price-space path under weak active liquidity.
        frame["flow_localisation_v4"] = (
            frame["swap_flow_intensity"]
            / frame["rho"].clip(lower=0.10)
            / (1.0 + frame["spatial_width_ratio_v4"])
        )

        # Spatial strain is retained as a diagnostic: large flow producing a
        # wide tick excursion under weak liquidity.
        frame["spatial_strain_v4"] = (
            frame["swap_flow_intensity"]
            * frame["spatial_width_ratio_v4"]
            / frame["rho"].clip(lower=0.10)
        )

        # Energy proxy. Unlike physical kinetic energy this is dimensionless
        # and uses realised tick-path width as the local spatial scale.
        frame["energy_proxy_v4"] = (
            frame["rho"].clip(lower=0.10)
            * frame["spatial_width_ratio_v4"]
            * frame["u"].pow(2)
        )

        frame["z_localisation_v4"] = causal_z(np.log1p(frame["flow_localisation_v4"]))
        frame["z_spatial_strain_v4"] = causal_z(np.log1p(frame["spatial_strain_v4"]))
        frame["z_energy_v4"] = causal_z(np.log1p(frame["energy_proxy_v4"]))
        frame["z_cancellation_v4"] = causal_z(np.log1p(frame["cancellation_stress"]))

        # Core V4 state: spatially localised flow + positive nonlinear
        # amplification + liquidity deficit + large hidden-term cancellation.
        # Equal weights are predeclared; no 12-month fit is performed.
        frame["instability_v4"] = (
            frame["z_localisation_v4"]
            + frame["z_amplification_v2"]
            + frame["z_liquidity_deficit_v2"]
            + frame["z_cancellation_v4"]
        ) / 4.0

        frame["instability_v4_threshold"] = (
            frame["instability_v4"]
            .rolling(168, min_periods=48)
            .quantile(0.90)
            .shift(1)
        )
        frame["high_instability_v4"] = (
            frame["instability_v4"] > frame["instability_v4_threshold"]
        )

        # V4-confirmed state: preserve the stronger V2 detector and use the
        # OpenAI-inspired spatial terms only as confirmation.
        # "Pre-critical" means V2 instability is high, localisation/cancellation
        # are above their causal baselines, while the energy proxy has not yet
        # become an extreme (> 2 causal standard deviations).
        frame["spatial_confirmation_v4"] = (
            frame["z_localisation_v4"] + frame["z_cancellation_v4"]
        ) / 2.0
        frame["high_instability_v4_confirmed"] = (
            frame["high_instability_v2"]
            & (frame["spatial_confirmation_v4"] > 0.0)
            & (frame["z_energy_v4"] < 2.0)
        )

        # Same damped integration spirit as the paper, but fully causal.
        frame["u_predicted_next"] = (
            frame["u"] + req.integration_gain * frame["net_force"]
        )
        frame["signal_for_next_hour"] = np.sign(frame["u_predicted_next"])
        frame["next_log_ret"] = frame["log_ret"].shift(-1)
        frame["next_abs_log_ret"] = frame["next_log_ret"].abs()
        frame["next_rv3"] = np.sqrt(
            sum(frame["log_ret"].shift(-k).pow(2) for k in range(1, 4))
        )
        frame["next_rv6"] = np.sqrt(
            sum(frame["log_ret"].shift(-k).pow(2) for k in range(1, 7))
        )
        frame["large_move_cutoff"] = (
            frame["log_ret"].abs()
            .rolling(168, min_periods=48)
            .quantile(0.75)
        )
        frame["next_hour_large_move"] = (
            frame["next_abs_log_ret"] > frame["large_move_cutoff"]
        )

        frame["position"] = frame["signal_for_next_hour"].shift(1).fillna(0.0)

        # Stress-gated directional probes. The detector decides WHEN to trade;
        # direction is deliberately kept as three simple, predeclared rules.
        high = frame["high_instability_v2"].fillna(False)
        frame["stress_momentum_signal"] = np.where(high, np.sign(frame["log_ret"]), 0.0)
        frame["stress_reversal_signal"] = np.where(high, -np.sign(frame["log_ret"]), 0.0)
        frame["stress_ns_signal"] = np.where(high, np.sign(frame["u_predicted_next"]), 0.0)
        frame["stress_momentum_position"] = frame["stress_momentum_signal"].shift(1).fillna(0.0)
        frame["stress_reversal_position"] = frame["stress_reversal_signal"].shift(1).fillna(0.0)
        frame["stress_ns_position"] = frame["stress_ns_signal"].shift(1).fillna(0.0)

        # V3: use the detector as an event trigger instead of trading every
        # stressed hour. A new high-instability regime takes the just-completed
        # hour's momentum direction and holds it for the NEXT 3 hours.
        # This horizon was chosen before the 12-month test because both July
        # and August V2 showed persistence in the 3h magnitude signal.
        rising_edge = high & (~high.shift(1).fillna(False))
        event_pos = np.zeros(len(frame), dtype=float)
        event_trigger = np.zeros(len(frame), dtype=bool)
        for i in range(len(frame) - 1):
            if not bool(rising_edge.iloc[i]):
                continue
            direction = float(np.sign(frame["log_ret"].iloc[i]))
            if direction == 0.0:
                continue
            start_i = i + 1
            end_i = min(i + 4, len(frame))
            if np.any(event_pos[start_i:end_i] != 0):
                continue
            event_pos[start_i:end_i] = direction
            event_trigger[i] = True

        frame["event_momentum_3h_position"] = event_pos
        frame["event_momentum_3h_trigger"] = event_trigger

        # V4 uses the same untouched execution rule as V3 so only the detector
        # changes: rising-edge V4 event -> completed-hour momentum -> next 3h.
        high_v4 = frame["high_instability_v4"].fillna(False)
        rising_edge_v4 = high_v4 & (~high_v4.shift(1).fillna(False))
        event_pos_v4 = np.zeros(len(frame), dtype=float)
        event_trigger_v4 = np.zeros(len(frame), dtype=bool)
        for i in range(len(frame) - 1):
            if not bool(rising_edge_v4.iloc[i]):
                continue
            direction = float(np.sign(frame["log_ret"].iloc[i]))
            if direction == 0.0:
                continue
            start_i = i + 1
            end_i = min(i + 4, len(frame))
            if np.any(event_pos_v4[start_i:end_i] != 0):
                continue
            event_pos_v4[start_i:end_i] = direction
            event_trigger_v4[i] = True

        frame["v4_event_momentum_3h_position"] = event_pos_v4
        frame["v4_event_momentum_3h_trigger"] = event_trigger_v4

        confirmed = frame["high_instability_v4_confirmed"].fillna(False)
        rising_edge_confirmed = confirmed & (~confirmed.shift(1).fillna(False))
        event_pos_confirmed = np.zeros(len(frame), dtype=float)
        event_trigger_confirmed = np.zeros(len(frame), dtype=bool)
        for i in range(len(frame) - 1):
            if not bool(rising_edge_confirmed.iloc[i]):
                continue
            direction = float(np.sign(frame["log_ret"].iloc[i]))
            if direction == 0.0:
                continue
            start_i = i + 1
            end_i = min(i + 4, len(frame))
            if np.any(event_pos_confirmed[start_i:end_i] != 0):
                continue
            event_pos_confirmed[start_i:end_i] = direction
            event_trigger_confirmed[i] = True

        frame["v4_confirmed_momentum_3h_position"] = event_pos_confirmed
        frame["v4_confirmed_momentum_3h_trigger"] = event_trigger_confirmed

        # Predeclared V3 momentum-continuation research gates.
        # These are locked before inspecting the prior 12-month holdout.
        abs_ret_median = (
            frame["log_ret"].abs().rolling(72, min_periods=24).median().shift(1)
        )
        momentum_sign = np.sign(frame["log_ret"])
        ns_agreement = np.sign(frame["u_predicted_next"]) == momentum_sign

        candidate_conditions = {
            "amp_positive": frame["z_amplification_v2"] > 0.0,
            "flow_positive": frame["z_flow_pressure_v2"] > 0.0,
            "liq_deficit_positive": frame["z_liquidity_deficit_v2"] > 0.0,
            "ns_agrees_momentum": ns_agreement,
            "momentum_strong": frame["log_ret"].abs() > abs_ret_median,
            "flow_and_amp_positive": (
                (frame["z_flow_pressure_v2"] > 0.0)
                & (frame["z_amplification_v2"] > 0.0)
            ),
            "localisation_positive": frame["z_localisation_v4"] > 0.0,
            "energy_not_extreme": frame["z_energy_v4"] < 2.0,
            "cancellation_positive": frame["z_cancellation_v4"] > 0.0,
        }

        def _build_event_momentum(trigger: pd.Series, hold_hours: int = 3):
            pos = np.zeros(len(frame), dtype=float)
            trig = np.zeros(len(frame), dtype=bool)
            trigger = trigger.fillna(False).astype(bool)
            for i in range(len(frame) - 1):
                if not bool(trigger.iloc[i]):
                    continue
                direction = float(np.sign(frame["log_ret"].iloc[i]))
                if direction == 0.0:
                    continue
                start_i = i + 1
                end_i = min(i + 1 + hold_hours, len(frame))
                if np.any(pos[start_i:end_i] != 0):
                    continue
                pos[start_i:end_i] = direction
                trig[i] = True
            return pos, trig

        candidate_prefixes = []
        for name, cond in candidate_conditions.items():
            pos_c, trig_c = _build_event_momentum(rising_edge & cond, hold_hours=3)
            prefix = f"v3gate_{name}"
            frame[f"{prefix}_position"] = pos_c
            frame[f"{prefix}_trigger"] = trig_c
            candidate_prefixes.append(prefix)

        frame["gross_strategy_ret"] = frame["position"] * frame["log_ret"]

        previous_position = frame["position"].shift(1).fillna(0.0)
        frame["turnover"] = (frame["position"] - previous_position).abs()
        cost_rate = req.illustrative_cost_bps_per_turnover / 10_000.0
        frame["net_strategy_ret"] = (
            (1.0 + frame["gross_strategy_ret"])
            * (1.0 - cost_rate * frame["turnover"])
            - 1.0
        )

        def _apply_probe(position_col: str, prefix: str):
            p = frame[position_col]
            prev = p.shift(1).fillna(0.0)
            turn = (p - prev).abs()
            signed_log_ret = p * frame["log_ret"]
            gross_simple = np.expm1(signed_log_ret)
            net = np.exp(signed_log_ret) * (1.0 - cost_rate * turn) - 1.0
            frame[f"{prefix}_signed_log_ret"] = signed_log_ret
            frame[f"{prefix}_gross_ret"] = gross_simple
            frame[f"{prefix}_net_ret"] = net
            frame[f"{prefix}_turnover"] = turn

        _apply_probe("stress_momentum_position", "stress_momentum")
        _apply_probe("stress_reversal_position", "stress_reversal")
        _apply_probe("stress_ns_position", "stress_ns")
        _apply_probe("event_momentum_3h_position", "event_momentum_3h")
        _apply_probe("v4_event_momentum_3h_position", "v4_event_momentum_3h")
        _apply_probe("v4_confirmed_momentum_3h_position", "v4_confirmed_momentum_3h")
        for prefix in candidate_prefixes:
            _apply_probe(f"{prefix}_position", prefix)

        day = frame[(frame.index >= target) & (frame.index < target_end)].copy()
        needed = [
            "u", "rho", "A_advection", "P_pressure", "D_diffusion",
            "net_force", "flow_to_liquidity", "amplification_minus_damping",
            "cancellation_stress", "criticality", "u_predicted_next",
            "instability_v2", "instability_v2_threshold",
            "instability_v4", "instability_v4_threshold",
            "position", "log_ret", "gross_strategy_ret", "net_strategy_ret",
        ]
        day = day.dropna(subset=needed)
        expected_hours = int(req.days * 24)
        if len(day) < max(20, int(expected_hours * 0.90)):
            raise RuntimeError(
                f"Only {len(day)} complete experimental hours for {expected_hours} expected hours"
            )

        actual_sign = np.sign(day["log_ret"].to_numpy())
        pos = day["position"].to_numpy()
        active = pos != 0
        direction_accuracy = (
            float(np.mean(pos[active] == actual_sign[active]) * 100.0)
            if active.any()
            else 0.0
        )
        gross_total = float(np.exp(day["gross_strategy_ret"].sum()) - 1.0)
        net_total = float(np.prod(1.0 + day["net_strategy_ret"]) - 1.0)
        buy_hold = float(np.exp(day["log_ret"].sum()) - 1.0)

        predictive = day.dropna(subset=["next_abs_log_ret"]).copy()
        def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
            if len(a) < 3 or a.std() == 0 or b.std() == 0:
                return None
            value = a.corr(b)
            return float(value) if pd.notna(value) else None

        high_stress = predictive["criticality"] >= 1.0
        high_stress_count = int(high_stress.sum())
        high_stress_next_abs = (
            float(predictive.loc[high_stress, "next_abs_log_ret"].mean() * 100.0)
            if high_stress_count
            else None
        )
        normal_stress_next_abs = (
            float(predictive.loc[~high_stress, "next_abs_log_ret"].mean() * 100.0)
            if (~high_stress).any()
            else None
        )
        high_dir_accuracy = None
        if high_stress_count:
            predicted_dir = np.sign(
                predictive.loc[high_stress, "u_predicted_next"].to_numpy()
            )
            actual_next_dir = np.sign(
                predictive.loc[high_stress, "next_log_ret"].to_numpy()
            )
            high_dir_accuracy = float(
                np.mean(predicted_dir == actual_next_dir) * 100.0
            )

        # V2 magnitude validation
        predictive_v2 = day.dropna(
            subset=[
                "instability_v2", "instability_v2_threshold",
                "next_abs_log_ret", "next_rv3", "next_rv6",
                "large_move_cutoff",
            ]
        ).copy()
        high_v2 = predictive_v2["high_instability_v2"].astype(bool)
        n_high_v2 = int(high_v2.sum())
        n_low_v2 = int((~high_v2).sum())

        def _mean_pct(mask: pd.Series, col: str):
            if int(mask.sum()) == 0:
                return None
            return float(predictive_v2.loc[mask, col].mean() * 100.0)

        high_abs1 = _mean_pct(high_v2, "next_abs_log_ret")
        low_abs1 = _mean_pct(~high_v2, "next_abs_log_ret")
        high_rv3 = _mean_pct(high_v2, "next_rv3")
        low_rv3 = _mean_pct(~high_v2, "next_rv3")
        high_rv6 = _mean_pct(high_v2, "next_rv6")
        low_rv6 = _mean_pct(~high_v2, "next_rv6")
        large_move_precision = (
            float(predictive_v2.loc[high_v2, "next_hour_large_move"].mean() * 100.0)
            if n_high_v2 else None
        )
        large_move_baseline = float(
            predictive_v2["next_hour_large_move"].mean() * 100.0
        ) if len(predictive_v2) else None

        predictive_v4 = day.dropna(
            subset=[
                "instability_v4", "instability_v4_threshold",
                "next_abs_log_ret", "next_rv3", "next_rv6",
                "large_move_cutoff",
            ]
        ).copy()
        high_v4_eval = predictive_v4["high_instability_v4"].astype(bool)
        n_high_v4 = int(high_v4_eval.sum())
        n_low_v4 = int((~high_v4_eval).sum())

        def _mean_pct_v4(mask: pd.Series, col: str):
            if int(mask.sum()) == 0:
                return None
            return float(predictive_v4.loc[mask, col].mean() * 100.0)

        v4_high_abs1 = _mean_pct_v4(high_v4_eval, "next_abs_log_ret")
        v4_low_abs1 = _mean_pct_v4(~high_v4_eval, "next_abs_log_ret")
        v4_high_rv3 = _mean_pct_v4(high_v4_eval, "next_rv3")
        v4_low_rv3 = _mean_pct_v4(~high_v4_eval, "next_rv3")
        v4_high_rv6 = _mean_pct_v4(high_v4_eval, "next_rv6")
        v4_low_rv6 = _mean_pct_v4(~high_v4_eval, "next_rv6")
        v4_large_move_precision = (
            float(predictive_v4.loc[high_v4_eval, "next_hour_large_move"].mean() * 100.0)
            if n_high_v4 else None
        )
        v4_large_move_baseline = (
            float(predictive_v4["next_hour_large_move"].mean() * 100.0)
            if len(predictive_v4) else None
        )

        def _lift(high_value, low_value):
            if high_value is None or low_value in (None, 0):
                return None
            return float(high_value / low_value)

        def _probe_result(prefix: str):
            g = day.dropna(subset=[f"{prefix}_net_ret", f"{prefix}_gross_ret"])
            gross_curve = (1.0 + g[f"{prefix}_gross_ret"]).cumprod()
            gross = float(gross_curve.iloc[-1] - 1.0) if len(gross_curve) else 0.0
            net_curve = (1.0 + g[f"{prefix}_net_ret"]).cumprod()
            net = float(net_curve.iloc[-1] - 1.0) if len(net_curve) else 0.0
            peak = net_curve.cummax()
            drawdown = net_curve / peak - 1.0
            position_col = f"{prefix}_position"
            return {
                "gross_return_pct": gross * 100.0,
                "net_return_pct": net * 100.0,
                "max_drawdown_pct": float(drawdown.min() * 100.0) if len(drawdown) else 0.0,
                "total_turnover": float(g[f"{prefix}_turnover"].sum()),
                "active_position_hours": (
                    int((day[position_col] != 0).sum())
                    if position_col in day.columns else None
                ),
                "turnover_hours": int((day[f"{prefix}_turnover"] > 0).sum()),
            }

        def _probe_cost_sensitivity(prefix: str):
            g = day.dropna(subset=[f"{prefix}_signed_log_ret", f"{prefix}_turnover"])
            signed = g[f"{prefix}_signed_log_ret"]
            turn = g[f"{prefix}_turnover"]
            out = {}
            for bps in (0, 2, 5, 10, 15, 20):
                cr = bps / 10_000.0
                step = np.exp(signed) * (1.0 - cr * turn) - 1.0
                curve = (1.0 + step).cumprod()
                peak = curve.cummax()
                dd = curve / peak - 1.0
                out[str(bps)] = {
                    "net_return_pct": float((curve.iloc[-1] - 1.0) * 100.0) if len(curve) else 0.0,
                    "max_drawdown_pct": float(dd.min() * 100.0) if len(dd) else 0.0,
                }
            return out

        daily_rows = []
        for trading_date, g in day.groupby(day.index.floor("D")):
            g_active = g["position"].to_numpy() != 0
            g_actual_sign = np.sign(g["log_ret"].to_numpy())
            g_pos = g["position"].to_numpy()
            g_dir_acc = (
                float(np.mean(g_pos[g_active] == g_actual_sign[g_active]) * 100.0)
                if g_active.any()
                else 0.0
            )
            daily_rows.append({
                "date": str(pd.Timestamp(trading_date).date()),
                "hours": int(len(g)),
                "buy_hold_return_pct": float((np.exp(g["log_ret"].sum()) - 1.0) * 100.0),
                "gross_strategy_return_pct": float((np.exp(g["gross_strategy_ret"].sum()) - 1.0) * 100.0),
                "net_strategy_return_pct": float((np.prod(1.0 + g["net_strategy_ret"]) - 1.0) * 100.0),
                "direction_accuracy_pct": g_dir_acc,
                "turnover": float(g["turnover"].sum()),
                "mean_criticality": float(g["criticality"].mean()),
                "max_criticality": float(g["criticality"].max()),
            })

        profitable_days = int(sum(1 for row in daily_rows if row["net_strategy_return_pct"] > 0))

        monthly_rows = []
        for month, g in day.groupby(day.index.to_period("M")):
            v2_curve = (1.0 + g["stress_momentum_net_ret"]).cumprod()
            v3_curve = (1.0 + g["event_momentum_3h_net_ret"]).cumprod()
            v4_curve = (1.0 + g["v4_event_momentum_3h_net_ret"]).cumprod()
            v4c_curve = (1.0 + g["v4_confirmed_momentum_3h_net_ret"]).cumprod()
            monthly_rows.append({
                "month": str(month),
                "hours": int(len(g)),
                "buy_hold_return_pct": float((np.exp(g["log_ret"].sum()) - 1.0) * 100.0),
                "v2_stress_momentum_net_return_pct": (
                    float((v2_curve.iloc[-1] - 1.0) * 100.0) if len(v2_curve) else 0.0
                ),
                "v3_event_momentum_3h_net_return_pct": (
                    float((v3_curve.iloc[-1] - 1.0) * 100.0) if len(v3_curve) else 0.0
                ),
                "v4_event_momentum_3h_net_return_pct": (
                    float((v4_curve.iloc[-1] - 1.0) * 100.0) if len(v4_curve) else 0.0
                ),
                "v4_confirmed_momentum_3h_net_return_pct": (
                    float((v4c_curve.iloc[-1] - 1.0) * 100.0) if len(v4c_curve) else 0.0
                ),
                "v3_turnover": float(g["event_momentum_3h_turnover"].sum()),
                "v3_event_triggers": int(g["event_momentum_3h_trigger"].sum()),
                "v4_turnover": float(g["v4_event_momentum_3h_turnover"].sum()),
                "v4_event_triggers": int(g["v4_event_momentum_3h_trigger"].sum()),
                "v4_confirmed_turnover": float(g["v4_confirmed_momentum_3h_turnover"].sum()),
                "v4_confirmed_event_triggers": int(g["v4_confirmed_momentum_3h_trigger"].sum()),
            })

        top = (
            day.sort_values("criticality", ascending=False)
            .head(10)
            .sort_index()
        )
        top_events = []
        for ts, row in top.iterrows():
            top_events.append({
                "timestamp": str(ts),
                "criticality": float(row["criticality"]),
                "u": float(row["u"]),
                "rho": float(row["rho"]),
                "flow_to_liquidity": float(row["flow_to_liquidity"]),
                "amplification_minus_damping": float(row["amplification_minus_damping"]),
                "cancellation_stress": float(row["cancellation_stress"]),
                "net_force": float(row["net_force"]),
                "position_during_hour": int(row["position"]),
                "hour_return_pct": float((math.exp(row["log_ret"]) - 1.0) * 100.0),
            })

        hourly = []
        for ts, row in day.iterrows():
            hourly.append({
                "timestamp": str(ts),
                "price": float(row["close"]),
                "u": float(row["u"]),
                "rho": float(row["rho"]),
                "A": float(row["A_advection"]),
                "P": float(row["P_pressure"]),
                "D": float(row["D_diffusion"]),
                "net_force": float(row["net_force"]),
                "criticality": float(row["criticality"]),
                "cancellation_stress": float(row["cancellation_stress"]),
                "signal_for_next_hour": int(row["signal_for_next_hour"]),
                "position": int(row["position"]),
                "hour_return_pct": float((math.exp(row["log_ret"]) - 1.0) * 100.0),
            })

        return {
            "status": "ok",
            "experiment": "Navier-Stokes-inspired endogenous criticality prototype",
            "date": str(target.date()),
            "market": {
                "chain": "ethereum",
                "exchange": pair.exchange_slug,
                "pair_id": int(pair.pair_id),
                "pair": f"{pair.base_token_symbol}/{pair.quote_token_symbol}",
                "fee_tier": float(pair.fee_tier),
                "price_bucket": "1h",
                "liquidity_bucket": "1m aggregated to 1h",
            },
            "causality": {
                "lookahead": False,
                "rule": "features at hour t determine position for hour t+1",
                "warmup_hours": 72,
            },
            "mapping": {
                "u": "volume-weighted normalized log-return velocity",
                "rho": "active CLMM liquidity / trailing 24h median",
                "p": "log price displacement from trailing 6h volume-weighted price",
                "nu": "6h realized volatility / 24h realized volatility",
                "external_force": "omitted in this first endogenous-only prototype",
                "concentration": "|u| / rho (flow-to-active-liquidity stress)",
                "amplification": "|A+P| - |D|",
                "cancellation": "(|A|+|P|+|D|)/(|A+P+D|+epsilon)",
            },
            "result": {
                "requested_days": int(req.days),
                "expected_hours": expected_hours,
                "complete_hours": int(len(day)),
                "coverage_pct": float(len(day) / expected_hours * 100.0),
                "start_price": float(day["close"].iloc[0]),
                "end_price": float(day["close"].iloc[-1]),
                "buy_hold_return_pct": buy_hold * 100.0,
                "direction_accuracy_pct": direction_accuracy,
                "gross_strategy_return_pct": gross_total * 100.0,
                "net_strategy_return_pct": net_total * 100.0,
                "illustrative_cost_bps_per_turnover": req.illustrative_cost_bps_per_turnover,
                "total_turnover": float(day["turnover"].sum()),
                "profitable_days_net": profitable_days,
                "losing_or_flat_days_net": int(len(daily_rows) - profitable_days),
                "mean_criticality": float(day["criticality"].mean()),
                "max_criticality": float(day["criticality"].max()),
                "max_cancellation_stress": float(day["cancellation_stress"].max()),
            },
            "next_hour_stress_test": {
                "criticality_vs_next_abs_return_corr": _safe_corr(
                    predictive["criticality"], predictive["next_abs_log_ret"]
                ),
                "flow_to_liquidity_vs_next_abs_return_corr": _safe_corr(
                    predictive["flow_to_liquidity"], predictive["next_abs_log_ret"]
                ),
                "cancellation_vs_next_abs_return_corr": _safe_corr(
                    predictive["cancellation_stress"], predictive["next_abs_log_ret"]
                ),
                "amplification_vs_next_abs_return_corr": _safe_corr(
                    predictive["amplification_minus_damping"], predictive["next_abs_log_ret"]
                ),
                "high_stress_threshold": 1.0,
                "high_stress_hours": high_stress_count,
                "mean_next_hour_abs_return_pct_when_high_stress": high_stress_next_abs,
                "mean_next_hour_abs_return_pct_otherwise": normal_stress_next_abs,
                "direction_accuracy_next_hour_when_high_stress_pct": high_dir_accuracy,
            },
            "instability_v2": {
                "design": (
                    "equal-weight causal z-scores of swap-flow/active-liquidity pressure, "
                    "positive nonlinear amplification over diffusion, and active-liquidity deficit; "
                    "stress threshold is trailing-7d 90th percentile shifted by one hour"
                ),
                "evaluated_hours": int(len(predictive_v2)),
                "high_instability_hours": n_high_v2,
                "other_hours": n_low_v2,
                "score_vs_next_abs_return_corr": _safe_corr(
                    predictive_v2["instability_v2"], predictive_v2["next_abs_log_ret"]
                ),
                "mean_next_1h_abs_return_pct_high": high_abs1,
                "mean_next_1h_abs_return_pct_other": low_abs1,
                "next_1h_magnitude_lift": _lift(high_abs1, low_abs1),
                "mean_next_3h_realized_magnitude_pct_high": high_rv3,
                "mean_next_3h_realized_magnitude_pct_other": low_rv3,
                "next_3h_magnitude_lift": _lift(high_rv3, low_rv3),
                "mean_next_6h_realized_magnitude_pct_high": high_rv6,
                "mean_next_6h_realized_magnitude_pct_other": low_rv6,
                "next_6h_magnitude_lift": _lift(high_rv6, low_rv6),
                "large_move_definition": "next |1h return| > causal trailing-7d 75th percentile",
                "large_move_precision_pct_when_high": large_move_precision,
                "large_move_unconditional_rate_pct": large_move_baseline,
                "large_move_precision_lift": _lift(
                    large_move_precision, large_move_baseline
                ),
            },
            "stress_gated_direction_probes": {
                "cost_bps_per_turnover": req.illustrative_cost_bps_per_turnover,
                "momentum": _probe_result("stress_momentum"),
                "reversal": _probe_result("stress_reversal"),
                "ns_direction": _probe_result("stress_ns"),
            },
            "v3_event_momentum_3h": {
                "logic": (
                    "rising-edge high-instability event; completed-hour momentum "
                    "sets direction for the next 3 hours; overlapping events ignored"
                ),
                "return_accounting": "exact signed log-return compounding before multiplicative turnover costs",
                "cost_bps_per_turnover": req.illustrative_cost_bps_per_turnover,
                **_probe_result("event_momentum_3h"),
                "cost_sensitivity_bps": _probe_cost_sensitivity("event_momentum_3h"),
                "event_triggers": int(day["event_momentum_3h_trigger"].sum()),
            },
            "v3_momentum_gate_research": {
                prefix.replace("v3gate_", ""): {
                    **_probe_result(prefix),
                    "cost_sensitivity_bps": _probe_cost_sensitivity(prefix),
                    "event_triggers": int(day[f"{prefix}_trigger"].sum()),
                }
                for prefix in candidate_prefixes
            },
            "instability_v4_spatial": {
                "design": (
                    "equal-weight causal z-scores of flow localisation in realised tick-space, "
                    "positive nonlinear amplification over diffusion, active-liquidity deficit, "
                    "and hidden-term cancellation; threshold is trailing-7d 90th percentile shifted 1h"
                ),
                "spatial_proxy_limitation": (
                    "tick-path dispersion is realised price-space traversal, not the full Uniswap "
                    "liquidity distribution across initialized ticks"
                ),
                "evaluated_hours": int(len(predictive_v4)),
                "high_instability_hours": n_high_v4,
                "other_hours": n_low_v4,
                "score_vs_next_abs_return_corr": _safe_corr(
                    predictive_v4["instability_v4"], predictive_v4["next_abs_log_ret"]
                ),
                "localisation_vs_next_abs_return_corr": _safe_corr(
                    predictive_v4["flow_localisation_v4"], predictive_v4["next_abs_log_ret"]
                ),
                "spatial_strain_vs_next_abs_return_corr": _safe_corr(
                    predictive_v4["spatial_strain_v4"], predictive_v4["next_abs_log_ret"]
                ),
                "energy_proxy_vs_next_abs_return_corr": _safe_corr(
                    predictive_v4["energy_proxy_v4"], predictive_v4["next_abs_log_ret"]
                ),
                "mean_next_1h_abs_return_pct_high": v4_high_abs1,
                "mean_next_1h_abs_return_pct_other": v4_low_abs1,
                "next_1h_magnitude_lift": _lift(v4_high_abs1, v4_low_abs1),
                "mean_next_3h_realized_magnitude_pct_high": v4_high_rv3,
                "mean_next_3h_realized_magnitude_pct_other": v4_low_rv3,
                "next_3h_magnitude_lift": _lift(v4_high_rv3, v4_low_rv3),
                "mean_next_6h_realized_magnitude_pct_high": v4_high_rv6,
                "mean_next_6h_realized_magnitude_pct_other": v4_low_rv6,
                "next_6h_magnitude_lift": _lift(v4_high_rv6, v4_low_rv6),
                "large_move_precision_pct_when_high": v4_large_move_precision,
                "large_move_unconditional_rate_pct": v4_large_move_baseline,
                "large_move_precision_lift": _lift(
                    v4_large_move_precision, v4_large_move_baseline
                ),
            },
            "v4_event_momentum_3h": {
                "logic": (
                    "rising-edge V4 spatial-instability event; completed-hour momentum "
                    "sets direction for the next 3 hours; overlapping events ignored"
                ),
                "cost_bps_per_turnover": req.illustrative_cost_bps_per_turnover,
                **_probe_result("v4_event_momentum_3h"),
                "event_triggers": int(day["v4_event_momentum_3h_trigger"].sum()),
            },
            "v4_confirmed_momentum_3h": {
                "logic": (
                    "V2 high-instability plus positive spatial localisation/cancellation confirmation "
                    "and non-extreme energy proxy; rising edge uses completed-hour momentum for next 3h"
                ),
                "cost_bps_per_turnover": req.illustrative_cost_bps_per_turnover,
                **_probe_result("v4_confirmed_momentum_3h"),
                "event_triggers": int(day["v4_confirmed_momentum_3h_trigger"].sum()),
            },
            "highest_criticality_hours": top_events,
            "daily": daily_rows,
            "monthly": monthly_rows,
            "hourly": hourly,
            "limitations": [
                "This is an exploratory financial analogy, not a physical Navier-Stokes solution.",
                "flow-to-liquidity uses active CLMM liquidity, not the full spatial liquidity-width distribution.",
                "short exposure is mathematical; borrowing/funding/liquidation are not modeled.",
                "the 5 bps cost is illustrative turnover cost and is not a full execution/slippage model.",
                "no parameters were optimized on the target day.",
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"N-S criticality experiment failed: {type(exc).__name__}: {exc}",
        ) from exc


@api.get("/selftest/ns-criticality-aug15")
def ns_criticality_aug15_selftest():
    return ns_criticality_day(NavierStokesDayRequest())


@api.get("/selftest/ns-criticality-august")
def ns_criticality_august_selftest():
    return ns_criticality_day(
        NavierStokesDayRequest(date="2026-08-01", days=31)
    )


@api.get("/selftest/ns-instability-july")
def ns_instability_july_selftest():
    return ns_criticality_day(
        NavierStokesDayRequest(date="2026-07-01", days=31)
    )


@api.get("/selftest/ns-instability-12m")
def ns_instability_12m_selftest():
    return ns_criticality_day(
        NavierStokesDayRequest(date="2025-09-01", days=365)
    )


@api.get("/selftest/ns-instability-v4-prior-12m")
def ns_instability_v4_prior_12m_selftest():
    return ns_criticality_day(
        NavierStokesDayRequest(date="2024-09-01", days=365)
    )


@api.get("/selftest/ns-instability-older-12m")
def ns_instability_older_12m_selftest():
    return ns_criticality_day(
        NavierStokesDayRequest(date="2023-09-01", days=366)
    )


@api.get("/selftest/ns-instability-older2-12m")
def ns_instability_older2_12m_selftest():
    return ns_criticality_day(
        NavierStokesDayRequest(date="2022-09-01", days=365)
    )


@api.post("/backtest/long-reversal")
def long_reversal_backtest(req: LongReversalBacktestRequest):
    """Run the study-like ETH Long/Reversal rule on our WETH/USDC 1h market.

    This intentionally mirrors the paper's core signal and turnover-cost model,
    while using a mathematical short instead of a margin/perpetual execution venue.
    """
    try:
        from tradingstrategy.chain import ChainId
        from tradingstrategy.timebucket import TimeBucket

        start = pd.Timestamp(_parse_dt(req.start))
        end = pd.Timestamp(_parse_dt(req.end))
        if end <= start:
            raise HTTPException(status_code=400, detail="end must be after start")
        if end - start > pd.Timedelta(days=3700):
            raise HTTPException(status_code=400, detail="Long/Reversal endpoint is limited to 3700 days per run")
        if end - start < pd.Timedelta(days=2):
            raise HTTPException(status_code=400, detail="Long/Reversal backtest requires at least 2 days")
        if not req.transaction_costs_bps:
            raise HTTPException(status_code=400, detail="transaction_costs_bps cannot be empty")
        if any(cost not in (0, 1, 2) for cost in req.transaction_costs_bps):
            raise HTTPException(
                status_code=400,
                detail="Study-like cost scenarios are limited to 0, 1, or 2 bps",
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

        # Two warm-up days ensure the previous trading-date row itself has
        # a complete night + day pair before we shift its daytime return.
        fetch_start = (start - pd.Timedelta(days=2)).to_pydatetime()
        fetch_end = end.to_pydatetime()
        candles = client.fetch_candles_by_pair_ids(
            [pair.pair_id],
            TimeBucket.h1,
            start_time=fetch_start,
            end_time=fetch_end,
            progress_bar_description="Long/Reversal WETH-USDC backtest",
        )
        if candles is None or len(candles) == 0:
            raise RuntimeError("Trading Strategy returned no WETH/USDC candles")

        x = candles.copy()
        if "timestamp" in x.columns:
            x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True).dt.tz_convert(None)
        elif isinstance(x.index, pd.DatetimeIndex):
            idx = pd.to_datetime(x.index, utc=True).tz_convert(None)
            x = x.copy()
            x["timestamp"] = idx
        else:
            raise RuntimeError("Candle dataset has no usable timestamp")

        x = x.reset_index(drop=True)
        raw_rows = int(len(x))
        duplicate_timestamps = int(x["timestamp"].duplicated().sum())
        x["close"] = pd.to_numeric(x["close"], errors="coerce")
        x = x[["timestamp", "close"]].dropna().sort_values("timestamp")
        if duplicate_timestamps:
            raise RuntimeError(f"Duplicate hourly timestamps in source data: {duplicate_timestamps}")
        x = x.drop_duplicates("timestamp", keep="last")
        x = x[x["close"] > 0].copy()
        if x.empty:
            raise RuntimeError("No valid positive-price candles after cleaning")

        actual_index = pd.DatetimeIndex(x["timestamp"])
        expected_index = pd.date_range(
            start=actual_index.min(),
            end=actual_index.max(),
            freq="1h",
        )
        missing_hours = int(len(expected_index.difference(actual_index)))
        extra_hours = int(len(actual_index.difference(expected_index)))
        off_grid_timestamps = int(
            (
                (x["timestamp"].dt.minute != 0)
                | (x["timestamp"].dt.second != 0)
                | (x["timestamp"].dt.microsecond != 0)
            ).sum()
        )
        data_quality = {
            "raw_rows": raw_rows,
            "rows_after_cleaning": int(len(x)),
            "duplicate_timestamps": duplicate_timestamps,
            "missing_hours": missing_hours,
            "extra_hours": extra_hours,
            "off_grid_timestamps": off_grid_timestamps,
            "data_start": str(actual_index.min()),
            "data_end": str(actual_index.max()),
            "hourly_continuity_ok": (
                missing_hours == 0
                and extra_hours == 0
                and off_grid_timestamps == 0
            ),
        }

        x["bar_ret"] = x["close"].pct_change()
        x = x.dropna(subset=["bar_ret"]).copy()

        day_start_hour = 5
        shifted = x["timestamp"] - pd.to_timedelta(day_start_hour, unit="h")
        x["cycle_start"] = shifted.dt.floor("D") + pd.to_timedelta(day_start_hour, unit="h")
        x["hours_since_cycle_start"] = (
            (x["timestamp"] - x["cycle_start"]) / pd.Timedelta(hours=1)
        ).astype(int)
        x["session"] = np.where(x["hours_since_cycle_start"] < 12, "day", "night")
        x["trading_date"] = x["cycle_start"].dt.floor("D")
        night_mask = x["session"].eq("night")
        x.loc[night_mask, "trading_date"] = (
            x.loc[night_mask, "cycle_start"] + pd.Timedelta(days=1)
        ).dt.floor("D")

        grouped = (
            x.groupby(["trading_date", "session"], observed=True)
            .agg(
                session_ret=("bar_ret", lambda values: (1.0 + values).prod() - 1.0),
                n_bars=("bar_ret", "size"),
                first_bar=("timestamp", "min"),
                last_bar=("timestamp", "max"),
            )
            .reset_index()
        )
        returns = grouped.pivot(index="trading_date", columns="session", values="session_ret")
        counts = grouped.pivot(index="trading_date", columns="session", values="n_bars")
        first_bars = grouped.pivot(index="trading_date", columns="session", values="first_bar")
        last_bars = grouped.pivot(index="trading_date", columns="session", values="last_bar")

        sessions = pd.DataFrame(index=returns.index).sort_index()
        sessions["night_ret"] = returns.get("night")
        sessions["day_ret"] = returns.get("day")
        sessions["n_night_bars"] = counts.get("night")
        sessions["n_day_bars"] = counts.get("day")
        sessions["night_first_bar"] = first_bars.get("night")
        sessions["night_last_bar"] = last_bars.get("night")
        sessions["day_first_bar"] = first_bars.get("day")
        sessions["day_last_bar"] = last_bars.get("day")
        sessions = sessions[
            sessions["night_ret"].notna()
            & sessions["day_ret"].notna()
            & sessions["n_night_bars"].eq(12)
            & sessions["n_day_bars"].eq(12)
        ].copy()

        # Critical anti-lookahead rule from the study: today's daytime reversal
        # sees only yesterday's daytime return. Never bridge a missing calendar
        # trading date: after a gap, the dynamic daytime signal is CASH.
        previous_day = sessions["day_ret"].shift(1)
        consecutive_day = sessions.index.to_series().diff().eq(pd.Timedelta(days=1))
        sessions["previous_day_ret"] = previous_day.where(consecutive_day, np.nan)
        sessions["night_position"] = 1.0
        sessions["day_position"] = -np.sign(sessions["previous_day_ret"].fillna(0.0))

        start_date = start.floor("D")
        end_date = end.floor("D")
        evaluated = sessions[
            (sessions.index >= start_date) & (sessions.index < end_date)
        ].copy()
        if evaluated.empty:
            raise RuntimeError("No complete 12h/12h trading dates in requested period")

        positions = np.column_stack(
            [
                evaluated["night_position"].to_numpy(dtype=float),
                evaluated["day_position"].to_numpy(dtype=float),
            ]
        ).reshape(-1)
        realized = np.column_stack(
            [
                evaluated["night_ret"].to_numpy(dtype=float),
                evaluated["day_ret"].to_numpy(dtype=float),
            ]
        ).reshape(-1)

        def _max_drawdown(returns_array: np.ndarray) -> float:
            wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns_array)))
            peaks = np.maximum.accumulate(wealth)
            return float(np.min(wealth / peaks - 1.0))

        scenarios = {}
        for cost_bps in sorted(set(req.transaction_costs_bps)):
            previous_position = np.concatenate(([0.0], positions[:-1]))
            turnover = np.abs(positions - previous_position)
            cost_rate = cost_bps / 10_000.0
            gross_returns = positions * realized
            net_returns = (1.0 + gross_returns) * (1.0 - cost_rate * turnover) - 1.0
            if np.any(1.0 + net_returns <= 0):
                raise RuntimeError("A session return reached -100% or below")

            wealth = req.initial_capital * np.cumprod(1.0 + net_returns)
            daily_returns = np.prod(1.0 + net_returns.reshape(-1, 2), axis=1) - 1.0
            daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
            sharpe = (
                float(np.mean(daily_returns) / daily_std * math.sqrt(365.0))
                if daily_std > 0
                else 0.0
            )

            scenarios[str(cost_bps)] = {
                "transaction_cost_bps": cost_bps,
                "final_value": float(wealth[-1]),
                "total_return_pct": float((wealth[-1] / req.initial_capital - 1.0) * 100.0),
                "max_drawdown_pct": float(_max_drawdown(net_returns) * 100.0),
                "sharpe_zero_rf": sharpe,
                "annualized_daily_volatility_pct": float(daily_std * math.sqrt(365.0) * 100.0),
                "position_changes": int(np.count_nonzero(turnover)),
                "total_turnover": float(turnover.sum()),
            }

        # Study-equivalent buy-and-hold is Long/Long over the same two
        # sessions, with the same turnover-cost formula. It pays only the
        # initial 0 -> +1 position change and then remains continuously long.
        benchmark_scenarios = {}
        benchmark_positions = np.ones_like(realized, dtype=float)
        benchmark_previous = np.concatenate(([0.0], benchmark_positions[:-1]))
        benchmark_turnover = np.abs(benchmark_positions - benchmark_previous)
        for cost_bps in sorted(set(req.transaction_costs_bps)):
            cost_rate = cost_bps / 10_000.0
            benchmark_net_returns = (
                (1.0 + realized) * (1.0 - cost_rate * benchmark_turnover) - 1.0
            )
            benchmark_wealth = req.initial_capital * np.cumprod(1.0 + benchmark_net_returns)
            benchmark_daily = (
                np.prod(1.0 + benchmark_net_returns.reshape(-1, 2), axis=1) - 1.0
            )
            benchmark_daily_std = (
                float(np.std(benchmark_daily, ddof=1))
                if len(benchmark_daily) > 1
                else 0.0
            )
            benchmark_sharpe = (
                float(
                    np.mean(benchmark_daily)
                    / benchmark_daily_std
                    * math.sqrt(365.0)
                )
                if benchmark_daily_std > 0
                else 0.0
            )
            benchmark_scenarios[str(cost_bps)] = {
                "transaction_cost_bps": cost_bps,
                "final_value": float(benchmark_wealth[-1]),
                "total_return_pct": float(
                    (benchmark_wealth[-1] / req.initial_capital - 1.0) * 100.0
                ),
                "max_drawdown_pct": float(
                    _max_drawdown(benchmark_net_returns) * 100.0
                ),
                "sharpe_zero_rf": benchmark_sharpe,
                "annualized_daily_volatility_pct": float(
                    benchmark_daily_std * math.sqrt(365.0) * 100.0
                ),
                "position_changes": int(np.count_nonzero(benchmark_turnover)),
                "total_turnover": float(benchmark_turnover.sum()),
            }

        rows = []
        for trading_date, row in evaluated.iterrows():
            rows.append(
                {
                    "trading_date": str(pd.Timestamp(trading_date).date()),
                    "previous_day_return_pct": float(row["previous_day_ret"] * 100.0),
                    "night_position": "LONG",
                    "night_return_pct": float(row["night_ret"] * 100.0),
                    "day_position": (
                        "SHORT" if row["day_position"] < 0
                        else "LONG" if row["day_position"] > 0
                        else "CASH"
                    ),
                    "day_return_pct": float(row["day_ret"] * 100.0),
                }
            )

        requested_dates = int((end_date - start_date) / pd.Timedelta(days=1))
        return {
            "status": "ok",
            "engine": "study-like vector backtest",
            "data_source": "Trading Strategy",
            "market": {
                "chain": "ethereum",
                "exchange": pair.exchange_slug,
                "pair_id": int(pair.pair_id),
                "base": pair.base_token_symbol,
                "quote": pair.quote_token_symbol,
                "fee_tier": float(pair.fee_tier),
                "time_bucket": "1h",
            },
            "strategy": {
                "name": "Long/Reversal (study-like)",
                "timezone": "UTC",
                "day_session": "05:00-17:00",
                "night_session": "17:00-05:00",
                "night_rule": "always LONG",
                "day_rule": "opposite sign of previous daytime session return",
                "first_requested_day_signal": (
                    "may use the immediately preceding same-session return, "
                    "matching the study holdout convention"
                ),
                "positions": {"long": 1, "cash": 0, "short": -1},
                "no_lookahead": True,
            },
            "request": req.model_dump(),
            "coverage": {
                "fetch_start": str(pd.Timestamp(fetch_start)),
                "fetch_end": str(pd.Timestamp(fetch_end)),
                "requested_trading_dates": requested_dates,
                "complete_trading_dates": int(len(evaluated)),
                "dropped_or_unavailable_dates": int(max(requested_dates - len(evaluated), 0)),
                "coverage_pct": float(len(evaluated) / requested_dates * 100.0),
                "first_trading_date_available": str(pd.Timestamp(evaluated.index.min()).date()),
                "last_trading_date_available": str(pd.Timestamp(evaluated.index.max()).date()),
                "first_session_bar": str(evaluated["night_first_bar"].min()),
                "last_session_bar": str(evaluated["day_last_bar"].max()),
                "gap_reset_to_cash_count": int(evaluated["previous_day_ret"].isna().sum()),
            },
            "data_quality": data_quality,
            "benchmark": {
                "name": "Buy-and-hold / Long-Long over the same session returns",
                "cost_scenarios": benchmark_scenarios,
            },
            "cost_scenarios": scenarios,
            "daily_sessions": rows,
            "adaptation_notes": {
                "matches_study_core": [
                    "1h data",
                    "UTC 12h/12h sessions",
                    "17:00-05:00 always long",
                    "05:00-17:00 reversal from previous same daytime session",
                    "positions +1/0/-1",
                    "0/1/2 bps turnover-cost scenarios",
                    "buy-and-hold Long/Long benchmark under the same turnover costs",
                    "explicit hourly continuity diagnostics",
                    "dynamic signal resets to cash after missing calendar trading dates",
                    "annualized daily volatility",
                    "no lookahead",
                ],
                "our_market": "WETH/USDC Uniswap V3 on Ethereum instead of ETH/USD Kraken",
                "simplified_short": "mathematical -1 exposure; no borrow, funding, margin or liquidation",
                "not_included": [
                    "slippage",
                    "bid-ask spread",
                    "gas",
                    "market impact",
                    "short financing/funding",
                ],
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Long/Reversal backtest failed: {type(exc).__name__}: {exc}",
        ) from exc


@api.get("/selftest/long-reversal")
def long_reversal_selftest():
    """Validate the study-like strategy on August 2026 WETH/USDC."""
    result = long_reversal_backtest(LongReversalBacktestRequest())
    if result["coverage"]["complete_trading_dates"] < 28:
        raise HTTPException(status_code=500, detail="Long/Reversal self-test has insufficient August coverage")
    return result


@api.get("/selftest/long-reversal-july")
def long_reversal_july_selftest():
    """Validate the study-like strategy on July 2026 WETH/USDC."""
    result = long_reversal_backtest(
        LongReversalBacktestRequest(
            start="2026-07-01T00:00:00Z",
            end="2026-08-01T00:00:00Z",
        )
    )
    if result["coverage"]["complete_trading_dates"] != 31:
        raise HTTPException(
            status_code=500,
            detail="Long/Reversal July self-test does not have all 31 trading dates",
        )
    return result


@api.get("/selftest/long-reversal-study-period")
def long_reversal_study_period_selftest():
    """Run the study's 2016-2025 calendar span on our WETH/USDC market."""
    return long_reversal_backtest(
        LongReversalBacktestRequest(
            start="2016-01-01T00:00:00Z",
            end="2026-01-01T00:00:00Z",
        )
    )


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


@api.post("/backtest/aave")
def aave_backtest(req: AaveBacktestRequest):
    """Run an Aave V3 Demeter simulation using Trading Strategy lending rates."""
    try:
        from decimal import Decimal
        from pathlib import Path as _Path

        from demeter import Actuator, AtTimeTrigger, MarketInfo, MarketTypeEnum, Strategy, TokenInfo
        from demeter.aave import AaveV3Market
        from tradingstrategy.chain import ChainId
        from tradingstrategy.lending import LendingCandleType, LendingProtocolType
        from tradingstrategy.timebucket import TimeBucket

        client = _get_ts_client()
        start = pd.Timestamp(_parse_dt(req.start))
        end = pd.Timestamp(_parse_dt(req.end))
        if end <= start:
            raise HTTPException(status_code=400, detail="end must be after start")
        if end - start < pd.Timedelta(hours=4):
            raise HTTPException(status_code=400, detail="Aave backtest requires at least 4 hours")
        if end - start > pd.Timedelta(days=31):
            raise HTTPException(status_code=400, detail="Aave backtest is limited to 31 days per run")
        if req.collateral_weth > req.initial_weth_balance:
            raise HTTPException(status_code=400, detail="collateral_weth exceeds initial_weth_balance")

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
            progress_bar_description="Aave backtest WETH price",
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
                market.supply(weth, req.collateral_weth, True)

            def _borrow(self, row_data):
                market.borrow(usdc, req.borrow_usdc)

            def _repay(self, row_data):
                for key in list(market.borrow_keys):
                    market.repay(key)

            def _withdraw(self, row_data):
                for key in list(market.supply_keys):
                    market.withdraw(key)

        actuator = Actuator()
        actuator.broker.add_market(market)
        actuator.broker.set_balance(weth, Decimal(str(req.initial_weth_balance)))
        actuator.broker.set_balance(usdc, Decimal(str(req.initial_usdc_balance)))
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
            "no_lookahead": True,
            "request": req.model_dump(),
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
            detail=f"Aave V3 backtest failed: {type(exc).__name__}: {exc}",
        ) from exc


@api.get("/selftest/aave")
def aave_selftest():
    """Run the default short Aave V3 integration smoke test."""
    return aave_backtest(AaveBacktestRequest())


@api.get("/selftest/native-backtest")
@api.get("/selftest/native_backtest")
def native_backtest_selftest():
    """Run the default native trade-executor + Trading Strategy backtest smoke test."""
    return native_backtest(NativeBacktestRequest())
