"""V8 reduced spatial Navier-Stokes-inspired market instability model.

The model is intentionally explicit about its limitation: Trading Strategy's
historical CLMM candles expose realised tick traversal, active liquidity and
swap-flow aggregates, but not the complete historical LP liquidity
distribution at every initialized tick. Therefore this module reconstructs a
causal *flow-pressure field along realised tick space*. It does not claim to
recover the true on-chain liquidity field rho(x,t).

Core structure:
    a_dot = B(a,a) - D a + f

B is parameterised with skew-symmetric nearest-neighbour generators so
    a.T @ B(a,a) == 0
by construction. D is diagonal and clipped non-negative.

A second-order mean/covariance closure is used:
    m_dot = B(m,m) + Bcov(C) - Dm + f
    C_dot = A(m)C + C A(m).T

The detector forecasts local effective intensity
    U_eff^2 = 2E / ell_eff
and signals concentration when forecast intensity grows while effective spatial
width contracts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class SpatialOperator:
    c: np.ndarray
    d: np.ndarray
    f: np.ndarray
    generators: list[np.ndarray]
    residual_rms: float
    samples: int


def _make_generators(n: int) -> list[np.ndarray]:
    """Nearest-neighbour skew generators; each preserves quadratic energy."""
    out: list[np.ndarray] = []
    for j in range(n - 1):
        k = np.zeros((n, n), dtype=float)
        k[j, j + 1] = 1.0
        k[j + 1, j] = -1.0
        out.append(k)
    return out


def _b(a: np.ndarray, op: SpatialOperator) -> np.ndarray:
    y = np.zeros_like(a, dtype=float)
    for j, (cj, kj) in enumerate(zip(op.c, op.generators)):
        y += cj * a[j] * (kj @ a)
    return y


def _bcov(cov: np.ndarray, op: SpatialOperator) -> np.ndarray:
    y = np.zeros(cov.shape[0], dtype=float)
    for j, (cj, kj) in enumerate(zip(op.c, op.generators)):
        y += cj * (kj @ cov[:, j])
    return y


def _linearised_a(m: np.ndarray, op: SpatialOperator) -> np.ndarray:
    n = len(m)
    out = -np.diag(op.d)
    for j, (cj, kj) in enumerate(zip(op.c, op.generators)):
        out = out + cj * m[j] * kj
        out[:, j] += cj * (kj @ m)
    return out


def _fit_operator(a_hist: np.ndarray, ridge: float = 0.05) -> SpatialOperator | None:
    """Fit structure-preserving reduced operator from past-only hourly fields."""
    a_hist = np.asarray(a_hist, dtype=float)
    a_hist = a_hist[np.all(np.isfinite(a_hist), axis=1)]
    if len(a_hist) < 96:
        return None

    n = a_hist.shape[1]
    gens = _make_generators(n)
    nc = len(gens)
    p = nc + n + n

    rows = []
    targets = []
    for x, y in zip(a_hist[:-1], a_hist[1:]):
        delta = y - x
        block = np.zeros((n, p), dtype=float)
        for j, kj in enumerate(gens):
            block[:, j] = x[j] * (kj @ x)
        for i in range(n):
            block[i, nc + i] = -x[i]
            block[i, nc + n + i] = 1.0
        rows.append(block)
        targets.append(delta)

    xmat = np.vstack(rows)
    yvec = np.concatenate(targets)
    reg = np.sqrt(ridge) * np.eye(p)
    beta = np.linalg.lstsq(
        np.vstack([xmat, reg]),
        np.concatenate([yvec, np.zeros(p)]),
        rcond=None,
    )[0]

    c = np.clip(beta[:nc], -2.0, 2.0)
    d = np.clip(beta[nc:nc+n], 0.0, 2.0)

    # Refit c and forcing after enforcing D >= 0.
    xcf_rows = []
    ycf = []
    for x, y in zip(a_hist[:-1], a_hist[1:]):
        delta = y - x + d * x
        block = np.zeros((n, nc + n), dtype=float)
        for j, kj in enumerate(gens):
            block[:, j] = x[j] * (kj @ x)
        for i in range(n):
            block[i, nc + i] = 1.0
        xcf_rows.append(block)
        ycf.append(delta)

    xcf = np.vstack(xcf_rows)
    ycf_vec = np.concatenate(ycf)
    reg2 = np.sqrt(ridge) * np.eye(nc + n)
    beta2 = np.linalg.lstsq(
        np.vstack([xcf, reg2]),
        np.concatenate([ycf_vec, np.zeros(nc+n)]),
        rcond=None,
    )[0]
    c = np.clip(beta2[:nc], -2.0, 2.0)
    f = np.clip(beta2[nc:], -1.0, 1.0)

    op = SpatialOperator(c=c, d=d, f=f, generators=gens, residual_rms=0.0, samples=len(a_hist)-1)
    preds = []
    actual = []
    for x, y in zip(a_hist[:-1], a_hist[1:]):
        preds.append(_b(x, op) - d * x + f)
        actual.append(y - x)
    residual = np.asarray(actual) - np.asarray(preds)
    op.residual_rms = float(np.sqrt(np.mean(residual ** 2)))
    return op


def _psd(c: np.ndarray) -> np.ndarray:
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    vals = np.clip(vals, 1e-8, 50.0)
    return (vecs * vals) @ vecs.T


def _energy_width(m: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    diag = np.clip(np.diag(c), 0.0, None)
    e = np.clip(m * m + diag, 0.0, None)
    total = float(np.sum(e))
    energy = 0.5 * total
    width = float((total * total) / (np.sum(e * e) + EPS)) if total > EPS else float(len(e))
    ueff2 = float(np.sum(e * e) / (total + EPS)) if total > EPS else 0.0
    return energy, width, ueff2


def _state_terms(m: np.ndarray, c: np.ndarray, op: SpatialOperator):
    bc = _bcov(c, op)
    bmm = _b(m, op)
    amat = _linearised_a(m, op)
    pi = float(-m @ bc)
    diss_cov = float(np.sum(op.d * np.diag(c)))
    trc = float(np.trace(c))
    lam = (pi - diss_cov) / (trc + EPS)
    return bmm, bc, amat, pi, diss_cov, lam


def _forecast(m0: np.ndarray, c0: np.ndarray, op: SpatialOperator, hours: int = 3):
    m = np.array(m0, dtype=float, copy=True)
    c = _psd(np.array(c0, dtype=float, copy=True))
    dt = 0.25
    steps = int(hours / dt)
    for _ in range(steps):
        bmm, bc, amat, _, _, _ = _state_terms(m, c, op)
        dm = bmm + bc - op.d * m + op.f
        dc = amat @ c + c @ amat.T
        m = np.clip(m + dt * dm, -12.0, 12.0)
        c = _psd(c + dt * dc)
    return m, c


def _orthonormal_dct_basis(n: int) -> np.ndarray:
    """DCT-II orthonormal basis: broad modes first, narrower modes later."""
    basis = np.zeros((n, n), dtype=float)
    for k in range(n):
        alpha = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        for j in range(n):
            basis[k, j] = alpha * math.cos(math.pi * (j + 0.5) * k / n)
    return basis


def _build_spatial_fields(q: pd.DataFrame, n_bins: int = 7) -> pd.DataFrame:
    q = q.copy()
    q["hour"] = q["timestamp"].dt.floor("h")

    # WETH/USDC pool has USDC as token0; caller can pre-map quote flow to this
    # canonical field name if a different pool is used later.
    signed = pd.to_numeric(q["signed_quote_flow"], errors="coerce").fillna(0.0)
    liq = pd.to_numeric(q["current_liquidity"], errors="coerce").abs()

    flow_scale = (
        signed.abs().rolling(360, min_periods=60).median().shift(1).replace(0, np.nan)
    )
    liq_scale = (
        liq.rolling(360, min_periods=60).median().shift(1).replace(0, np.nan)
    )
    signed_norm = (signed / flow_scale).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-20, 20)
    depth_norm = (liq / liq_scale).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.10, 10.0)
    q["impact"] = (signed_norm / depth_norm).clip(-20, 20)

    hour_open = q.groupby("hour")["open_tick"].first()
    hour_high = q.groupby("hour")["high_tick"].max()
    hour_low = q.groupby("hour")["low_tick"].min()
    hour_range = (hour_high - hour_low).abs().replace(0, np.nan)
    range_scale = hour_range.rolling(24, min_periods=8).median().shift(1)
    fallback = float(hour_range.dropna().median()) if hour_range.notna().any() else 10.0
    range_scale = range_scale.fillna(fallback).clip(lower=1.0)

    q = q.join(hour_open.rename("hour_open_tick"), on="hour")
    q = q.join(range_scale.rename("tick_scale"), on="hour")
    q["xi"] = ((q["close_tick"] - q["hour_open_tick"]) / q["tick_scale"]).clip(-3.499, 3.499)

    # Localized indicator basis on a co-moving tick coordinate.
    q["spatial_bin"] = np.floor((q["xi"] + 3.5) / (7.0 / n_bins)).astype(int)
    q["spatial_bin"] = q["spatial_bin"].clip(0, n_bins - 1)

    pivot = q.pivot_table(
        index="hour",
        columns="spatial_bin",
        values="impact",
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot = pivot.reindex(columns=list(range(n_bins)), fill_value=0.0)
    counts = q.groupby("hour").size().reindex(pivot.index).clip(lower=1)
    pivot = pivot.div(np.sqrt(counts), axis=0)
    pivot.columns = [f"x_bin_{i}" for i in range(n_bins)]

    # The selected model requires patterns at multiple spatial widths/scales.
    # Project the localized tick field into an orthonormal DCT basis:
    # low-order coefficients describe broad structures, high-order coefficients
    # describe progressively narrower oscillatory structures.
    basis = _orthonormal_dct_basis(n_bins)
    mode_values = pivot.to_numpy(dtype=float) @ basis.T
    modes = pd.DataFrame(
        mode_values,
        index=pivot.index,
        columns=[f"mode_raw_{i}" for i in range(n_bins)],
    )

    # Causal local standardisation makes the reduced coefficients comparable
    # across years without using current/future observations in the scale.
    hist_mean = modes.rolling(168, min_periods=48).mean().shift(1)
    hist_std = modes.rolling(168, min_periods=48).std().shift(1).replace(0, np.nan)
    z = ((modes - hist_mean) / hist_std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)

    # The reduced-order state should represent coherent structures rather than
    # raw minute noise. A 3-hour causal EWM is a filter, not a lookahead:
    # the coefficient at t uses observations only through t.
    z = z.ewm(span=3, adjust=False).mean()
    z.columns = [f"a_{i}" for i in range(n_bins)]
    return z


def _event_backtest(
    h: pd.DataFrame,
    trigger: pd.Series,
    direction: pd.Series,
    horizon: int,
    cost_bps: float,
):
    trigger = trigger.reindex(h.index).fillna(False).astype(bool)
    direction = direction.reindex(h.index).fillna(0.0)

    trades = []
    busy_until = -1
    idx = list(h.index)
    opens = h["open"].to_numpy(dtype=float)
    closes = h["close"].to_numpy(dtype=float)

    for i in range(len(h)):
        if i <= busy_until or not bool(trigger.iloc[i]):
            continue
        if i + horizon >= len(h):
            continue
        d = float(np.sign(direction.iloc[i]))
        if d == 0:
            continue
        entry_i = i + 1
        exit_i = i + horizon
        entry = float(opens[entry_i])
        exitp = float(closes[exit_i])
        if not (entry > 0 and exitp > 0):
            continue
        asset_ret = exitp / entry - 1.0
        gross = d * asset_ret
        net = gross - 2.0 * (cost_bps / 10_000.0) * abs(d)
        trades.append({
            "signal_time": idx[i],
            "entry_time": idx[entry_i],
            "exit_time": idx[exit_i],
            "direction": d,
            "entry": entry,
            "exit": exitp,
            "asset_return": asset_ret,
            "gross_return": gross,
            "net_return": net,
        })
        busy_until = exit_i

    wealth = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for tr in trades:
        wealth *= (1.0 + tr["net_return"])
        peak = max(peak, wealth)
        mdd = min(mdd, wealth / peak - 1.0)
        wins += int(tr["gross_return"] > 0)

    return {
        "trades": trades,
        "trade_count": len(trades),
        "gross_return_pct": (
            (np.prod([1.0 + t["gross_return"] for t in trades]) - 1.0) * 100.0
            if trades else 0.0
        ),
        "net_return_pct": (wealth - 1.0) * 100.0,
        "event_exit_max_drawdown_pct": mdd * 100.0,
        "win_rate_pct": (wins / len(trades) * 100.0) if trades else None,
        "roundtrip_turnover": float(2 * len(trades)),
        "active_hours": int(horizon * len(trades)),
    }


def run_v8_spatial(
    client,
    pair,
    target: pd.Timestamp,
    days: int,
    cost_bps: float = 5.0,
    horizon: int = 3,
    n_bins: int = 7,
):
    from tradingstrategy.timebucket import TimeBucket

    target = pd.Timestamp(target).floor("D")
    target_end = target + pd.Timedelta(days=days)
    # Operator identification needs substantially more history than V2.
    fetch_start = target - pd.Timedelta(days=35)
    fetch_end = target_end + pd.Timedelta(hours=horizon + 2)

    candles = client.fetch_candles_by_pair_ids(
        [pair.pair_id],
        TimeBucket.h1,
        start_time=fetch_start.to_pydatetime(),
        end_time=fetch_end.to_pydatetime(),
        progress_bar_description=f"V8 spatial {days}d OHLCV",
    )
    if candles is None or len(candles) == 0:
        raise RuntimeError("No V8 hourly OHLCV")

    h = candles.copy()
    if "timestamp" in h.columns:
        h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True).dt.tz_convert(None)
    elif isinstance(h.index, pd.DatetimeIndex):
        h["timestamp"] = pd.to_datetime(h.index, utc=True).tz_convert(None)
    else:
        raise RuntimeError("V8 hourly data has no timestamp")
    h = h.reset_index(drop=True)
    for col in ("open", "close", "volume"):
        h[col] = pd.to_numeric(h[col], errors="coerce")
    h = (
        h[["timestamp", "open", "close", "volume"]]
        .dropna()
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .set_index("timestamp")
    )
    h["log_ret"] = np.log(h["close"]).diff()
    h["next_abs_ret_1h"] = h["log_ret"].shift(-1).abs()
    h["next_rv3"] = np.sqrt(sum(h["log_ret"].shift(-k).pow(2) for k in range(1, 4)))

    parts = []
    chunk_start = fetch_start
    while chunk_start < fetch_end:
        chunk_end = min(chunk_start + pd.Timedelta(days=31), fetch_end)
        part = client.fetch_clmm_liquidity_provision_candles_by_pair_ids(
            [pair.pair_id],
            TimeBucket.m1,
            start_time=chunk_start.to_pydatetime(),
            end_time=chunk_end.to_pydatetime(),
            progress_bar_description=f"V8 CLMM {chunk_start.date()}",
        )
        if part is not None and len(part):
            parts.append(part)
        chunk_start = chunk_end
    if not parts:
        raise RuntimeError("No V8 CLMM data")

    q = pd.concat(parts, ignore_index=True)
    ts_col = "bucket" if "bucket" in q.columns else "timestamp"
    q["timestamp"] = pd.to_datetime(q[ts_col], utc=True).dt.tz_convert(None)
    q = q.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    required = [
        "open_tick", "close_tick", "high_tick", "low_tick",
        "current_liquidity", "net_amount0", "net_amount1",
    ]
    for col in required:
        if col not in q.columns:
            raise RuntimeError(f"V8 CLMM missing {col}")
        q[col] = pd.to_numeric(q[col], errors="coerce")

    # Positive signed quote flow means quote token enters the pool (buy-base pressure).
    if str(pair.token0_symbol).upper() == str(pair.quote_token_symbol).upper():
        q["signed_quote_flow"] = q["net_amount0"]
        quote_flow_source = "net_amount0"
    elif str(pair.token1_symbol).upper() == str(pair.quote_token_symbol).upper():
        q["signed_quote_flow"] = q["net_amount1"]
        quote_flow_source = "net_amount1"
    else:
        raise RuntimeError("Could not map quote token to CLMM amount column")

    fields = _build_spatial_fields(q, n_bins=n_bins)
    frame = h.join(fields, how="inner")
    acols = [f"a_{i}" for i in range(n_bins)]
    amat = frame[acols].to_numpy(dtype=float)

    score = np.full(len(frame), np.nan)
    lambda0 = np.full(len(frame), np.nan)
    ell_now = np.full(len(frame), np.nan)
    ell_hat = np.full(len(frame), np.nan)
    energy_now = np.full(len(frame), np.nan)
    energy_hat = np.full(len(frame), np.nan)
    pi_arr = np.full(len(frame), np.nan)
    diss_arr = np.full(len(frame), np.nan)
    fit_rms = np.full(len(frame), np.nan)

    op = None
    last_fit = -10_000
    fit_window = 720
    state_window = 72

    for i in range(len(frame)):
        if i < 240:
            continue

        if op is None or i - last_fit >= 24:
            hist = amat[max(0, i-fit_window):i]
            op = _fit_operator(hist)
            last_fit = i
        if op is None:
            continue

        state = amat[max(0, i-state_window+1):i+1]
        state = state[np.all(np.isfinite(state), axis=1)]
        if len(state) < 24:
            continue
        m = np.mean(state, axis=0)
        c = np.cov(state, rowvar=False)
        if np.ndim(c) != 2:
            continue
        c = _psd(c)

        e0, l0, _ = _energy_width(m, c)
        _, _, _, pi, diss, lam = _state_terms(m, c, op)
        mh, ch = _forecast(m, c, op, hours=horizon)
        eh, lh, _ = _energy_width(mh, ch)

        if e0 <= EPS or eh <= EPS or l0 <= EPS or lh <= EPS:
            continue
        sc = (1.0 / (2.0 * horizon)) * math.log((eh / e0) * (l0 / lh))

        score[i] = sc
        lambda0[i] = lam
        ell_now[i] = l0
        ell_hat[i] = lh
        energy_now[i] = e0
        energy_hat[i] = eh
        pi_arr[i] = pi
        diss_arr[i] = diss
        fit_rms[i] = op.residual_rms

    frame["v8_score"] = score
    frame["v8_lambda"] = lambda0
    frame["v8_ell"] = ell_now
    frame["v8_ell_hat"] = ell_hat
    frame["v8_energy"] = energy_now
    frame["v8_energy_hat"] = energy_hat
    frame["v8_pi"] = pi_arr
    frame["v8_diss_cov"] = diss_arr
    frame["v8_fit_rms"] = fit_rms

    frame["v8_concentration"] = (
        (frame["v8_score"] > 0.0)
        & (frame["v8_ell_hat"] < frame["v8_ell"])
    )
    frame["v8_internal_concentration"] = (
        frame["v8_concentration"]
        & (frame["v8_lambda"] > 0.0)
        & (frame["v8_pi"] > frame["v8_diss_cov"])
    )

    # A stricter causal extreme score variant; no future observations enter threshold.
    frame["v8_score_threshold"] = (
        frame["v8_score"].rolling(168, min_periods=48).quantile(0.90).shift(1)
    )
    frame["v8_extreme_concentration"] = (
        frame["v8_concentration"]
        & (frame["v8_score"] > frame["v8_score_threshold"])
    )

    # Rising edges only.
    for col in (
        "v8_concentration",
        "v8_internal_concentration",
        "v8_extreme_concentration",
    ):
        s = frame[col].fillna(False)
        frame[f"{col}_rising"] = s & (~s.shift(1).fillna(False))

    # Direction probes: old 1h momentum and orientation of the reconstructed field.
    frame["momentum_direction"] = np.sign(frame["log_ret"])
    frame["field_direction"] = np.sign(frame[acols].sum(axis=1))

    target_frame = frame[(frame.index >= target) & (frame.index < target_end)].copy()
    expected = days * 24
    if len(target_frame) < int(expected * 0.90):
        raise RuntimeError(f"V8 only has {len(target_frame)}/{expected} target hours")

    def mag_stats(signal_col: str):
        p = target_frame.dropna(subset=["next_abs_ret_1h", "next_rv3"]).copy()
        mask = p[signal_col].fillna(False).astype(bool)
        def mean(mask_, col):
            return float(p.loc[mask_, col].mean() * 100.0) if mask_.any() else None
        h1 = mean(mask, "next_abs_ret_1h")
        o1 = mean(~mask, "next_abs_ret_1h")
        h3 = mean(mask, "next_rv3")
        o3 = mean(~mask, "next_rv3")
        return {
            "signal_hours": int(mask.sum()),
            "other_hours": int((~mask).sum()),
            "next_1h_abs_return_pct_signal": h1,
            "next_1h_abs_return_pct_other": o1,
            "next_1h_lift": (h1 / o1) if h1 is not None and o1 not in (None, 0) else None,
            "next_3h_realized_magnitude_pct_signal": h3,
            "next_3h_realized_magnitude_pct_other": o3,
            "next_3h_lift": (h3 / o3) if h3 is not None and o3 not in (None, 0) else None,
        }

    probes = {}
    specs = {
        "concentration_momentum": ("v8_concentration_rising", "momentum_direction"),
        "concentration_field_direction": ("v8_concentration_rising", "field_direction"),
        "internal_momentum": ("v8_internal_concentration_rising", "momentum_direction"),
        "internal_field_direction": ("v8_internal_concentration_rising", "field_direction"),
        "extreme_momentum": ("v8_extreme_concentration_rising", "momentum_direction"),
        "extreme_field_direction": ("v8_extreme_concentration_rising", "field_direction"),
    }
    cost_grid = (0, 2, 5, 10, 15, 20)
    for name, (trig, direc) in specs.items():
        base = _event_backtest(target_frame, target_frame[trig], target_frame[direc], horizon, cost_bps)
        base["cost_sensitivity_bps"] = {
            str(c): {
                k: v
                for k, v in _event_backtest(
                    target_frame,
                    target_frame[trig],
                    target_frame[direc],
                    horizon,
                    c,
                ).items()
                if k != "trades"
            }
            for c in cost_grid
        }
        base.pop("trades", None)
        probes[name] = base

    valid = target_frame.dropna(subset=["v8_score", "next_abs_ret_1h"])
    corr = (
        float(valid["v8_score"].corr(valid["next_abs_ret_1h"]))
        if len(valid) >= 3 and valid["v8_score"].std() > 0 and valid["next_abs_ret_1h"].std() > 0
        else None
    )

    return {
        "status": "ok",
        "model": "V8 reduced spatial mean-covariance Navier-Stokes-inspired model",
        "period": {
            "start": str(target),
            "end": str(target_end),
            "days": int(days),
            "hours": int(len(target_frame)),
        },
        "causality": {
            "lookahead": False,
            "operator_fit": "rolling past 720h, refit every 24h",
            "state_estimate": "past/current 72h only",
            "forecast_horizon_hours": int(horizon),
            "execution": "rising-edge signal at t; entry next hour; exit after 3 hourly candles",
        },
        "spatial_field": {
            "bins": int(n_bins),
            "coordinate": "co-moving realised tick coordinate scaled by past 24h tick range",
            "signed_quote_flow_source": quote_flow_source,
            "depth_proxy": "active liquidity normalized by past minute liquidity",
            "basis": "localized tick field projected to an orthonormal multiscale DCT basis; 3h causal EWM state filter",
            "limitation": (
                "This reconstructs flow-pressure over realised tick traversal, not the full historical "
                "LP liquidity distribution across all initialized ticks."
            ),
        },
        "structure": {
            "equation": "a_dot = B(a,a) - D a + f",
            "energy_preserving_B": True,
            "nonnegative_diagonal_D": True,
            "closure": "second-order m/C closure; third moments omitted",
            "score": "(1/(2h))*log((Ehat/E)*(ell/ell_hat))",
        },
        "diagnostics": {
            "score_vs_next_abs_1h_corr": corr,
            "mean_fit_residual_rms": float(target_frame["v8_fit_rms"].mean()),
            "mean_lambda": float(target_frame["v8_lambda"].mean()),
            "mean_pi": float(target_frame["v8_pi"].mean()),
            "mean_cov_dissipation": float(target_frame["v8_diss_cov"].mean()),
            "score_positive_rate_pct": float((target_frame["v8_score"] > 0).mean() * 100.0),
            "width_contraction_rate_pct": float((target_frame["v8_ell_hat"] < target_frame["v8_ell"]).mean() * 100.0),
            "lambda_positive_rate_pct": float((target_frame["v8_lambda"] > 0).mean() * 100.0),
            "score_quantiles": {
                str(q): float(target_frame["v8_score"].quantile(q))
                for q in (0.01, 0.10, 0.50, 0.90, 0.99)
            },
            "lambda_quantiles": {
                str(q): float(target_frame["v8_lambda"].quantile(q))
                for q in (0.01, 0.10, 0.50, 0.90, 0.99)
            },
        },
        "magnitude_tests": {
            "concentration": mag_stats("v8_concentration"),
            "internal_concentration": mag_stats("v8_internal_concentration"),
            "extreme_concentration": mag_stats("v8_extreme_concentration"),
        },
        "strategy_probes_linear_short_accounting": probes,
        "return_accounting": (
            "Each event uses next-hour open as entry and close after the hold horizon as exit. "
            "Long/short P&L is linear: x*(P_exit/P_entry-1). Cost is charged on entry and exit."
        ),
    }
