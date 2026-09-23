"""Reduced spatial market model inspired by energy-preserving Navier-Stokes structure.

This module intentionally distinguishes between:
1) the mathematical structure of the reduced model, which enforces
   energy-preserving quadratic transfer and non-negative damping, and
2) the market observation layer, which is currently a proxy reconstructed from
   Trading Strategy minute CLMM candles.

Trading Strategy's CLMM schema exposes realised ticks, current active liquidity,
and aggregate swap-flow amounts, but not the full historical liquidity
distribution across all initialized ticks. Therefore this implementation is NOT
the final per-tick rho(x,t) model. It is the closest causal reduced-order model
supported by the currently available history.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


LOG_TICK = math.log(1.0001)


def _basis(z: np.ndarray) -> np.ndarray:
    """Five orthogonal multi-scale Fourier modes on z in [-1, 1]."""
    z = np.asarray(z, dtype=float)
    return np.column_stack(
        [
            np.ones_like(z) / math.sqrt(2.0),
            np.sin(math.pi * z),
            np.cos(math.pi * z),
            np.sin(2.0 * math.pi * z),
            np.cos(2.0 * math.pi * z),
        ]
    )


def _skew_basis(n: int):
    pairs = []
    mats = []
    for p in range(n):
        for q in range(p + 1, n):
            s = np.zeros((n, n), dtype=float)
            s[p, q] = 1.0
            s[q, p] = -1.0
            pairs.append((p, q))
            mats.append(s)
    return pairs, mats


@dataclass
class ReducedModel:
    S: list[np.ndarray]
    damping: np.ndarray
    forcing: np.ndarray

    def B(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mix = np.zeros((len(a), len(a)), dtype=float)
        for j, sj in enumerate(self.S):
            mix += a[j] * sj
        return mix @ b

    def drift(self, a: np.ndarray) -> np.ndarray:
        return self.B(a, a) - self.damping * a + self.forcing

    def jacobian(self, m: np.ndarray) -> np.ndarray:
        n = len(m)
        sm = np.zeros((n, n), dtype=float)
        cols = np.zeros((n, n), dtype=float)
        for j, sj in enumerate(self.S):
            sm += m[j] * sj
            cols[:, j] = sj @ m
        return sm + cols - np.diag(self.damping)

    def bbar_cov(self, c: np.ndarray) -> np.ndarray:
        out = np.zeros(c.shape[0], dtype=float)
        for j, sj in enumerate(self.S):
            out += sj @ c[j, :]
        return out


def _fit_reduced_model(a_hist: np.ndarray, ridge: float = 0.05) -> ReducedModel | None:
    """Fit a causal energy-preserving quadratic ROM on past hourly coefficients.

    B(a,a) = (sum_j a_j S_j) a, S_j skew-symmetric.
    D is diagonal and clamped non-negative after ridge fitting.
    """
    a_hist = np.asarray(a_hist, dtype=float)
    if a_hist.ndim != 2 or len(a_hist) < 96:
        return None

    n = a_hist.shape[1]
    _, skew_templates = _skew_basis(n)
    n_skew = len(skew_templates)
    n_theta = n * n_skew
    n_params = n_theta + n + n

    rows = []
    targets = []
    for t in range(len(a_hist) - 1):
        a = a_hist[t]
        y = a_hist[t + 1] - a
        if not (np.isfinite(a).all() and np.isfinite(y).all()):
            continue
        x = np.zeros((n, n_params), dtype=float)
        col = 0
        for j in range(n):
            for s in skew_templates:
                x[:, col] = a[j] * (s @ a)
                col += 1
        for i in range(n):
            x[i, n_theta + i] = -a[i]
            x[i, n_theta + n + i] = 1.0
        rows.append(x)
        targets.append(y)

    if len(rows) < 72:
        return None

    X = np.vstack(rows)
    y = np.concatenate(targets)
    reg = np.eye(n_params, dtype=float) * ridge
    # Forcing should not be punished as strongly as interaction coefficients.
    reg[n_theta + n :, n_theta + n :] *= 0.1
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y)

    s_mats = [np.zeros((n, n), dtype=float) for _ in range(n)]
    col = 0
    for j in range(n):
        for s in skew_templates:
            s_mats[j] += beta[col] * s
            col += 1

    damping = np.maximum(beta[n_theta : n_theta + n], 0.0)
    forcing = beta[n_theta + n : n_theta + 2 * n]
    return ReducedModel(S=s_mats, damping=damping, forcing=forcing)


def _psd(c: np.ndarray) -> np.ndarray:
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    vals = np.clip(vals, 1e-10, None)
    return (vecs * vals) @ vecs.T


def _forecast(model: ReducedModel, m: np.ndarray, c: np.ndarray, hours: int) -> tuple[np.ndarray, np.ndarray]:
    """Small-step causal forecast of mean and covariance."""
    m = np.asarray(m, dtype=float).copy()
    c = _psd(np.asarray(c, dtype=float))
    dt = 0.25
    steps = max(1, int(hours / dt))
    for _ in range(steps):
        A = model.jacobian(m)
        dm = model.drift(m)
        dc = A @ c + c @ A.T
        # Guard against numerical blow-ups while preserving sign/structure.
        dm = np.clip(dm, -5.0, 5.0)
        dc = np.clip(dc, -25.0, 25.0)
        m = np.clip(m + dt * dm, -20.0, 20.0)
        c = _psd(c + dt * dc)
        tr = float(np.trace(c))
        if tr > 1e4:
            c *= 1e4 / tr
    return m, c


def _energy_width(m: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    """Return total quadratic energy, effective width and local intensity."""
    n = len(m)
    zz = np.linspace(-1.0, 1.0, 257)
    psi = _basis(zz)[:, :n]
    q = np.outer(m, m) + c
    e = np.einsum("gi,ij,gj->g", psi, q, psi)
    e = np.clip(e, 0.0, None)

    int_e = float(np.trapz(e, zz))
    int_e2 = float(np.trapz(e * e, zz))
    energy = 0.5 * (float(m @ m) + float(np.trace(c)))
    if int_e <= 1e-12 or int_e2 <= 1e-12:
        return energy, 2.0, 0.0
    width = (int_e * int_e) / int_e2
    local_intensity = math.sqrt(max(2.0 * energy / max(width, 1e-12), 0.0))
    return energy, width, local_intensity


def _minute_activity_proxy(q: pd.DataFrame) -> pd.DataFrame:
    """Build a signed local activity proxy from currently available CLMM history.

    The sign comes from realised tick motion. Swap amount magnitude and active
    liquidity determine how much weight the minute receives. This is explicitly
    NOT equivalent to true signed per-tick order flow.
    """
    x = q.copy().sort_index()
    close_tick = pd.to_numeric(x["close_tick"], errors="coerce")
    liq = pd.to_numeric(x["current_liquidity"], errors="coerce").abs()

    tick_move = close_tick.diff().fillna(0.0)
    dlogp = tick_move * LOG_TICK

    liq_base = liq.rolling(1440, min_periods=120).median().shift(1)
    liq_rel = (liq / liq_base.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    liq_rel = liq_rel.fillna(1.0).clip(0.05, 20.0)

    mags = []
    for col in ("net_amount0", "net_amount1", "in_amount0", "in_amount1"):
        if col in x.columns:
            s = pd.to_numeric(x[col], errors="coerce").abs()
            med = s.rolling(1440, min_periods=120).median().shift(1)
            ratio = (s / med.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            mags.append(ratio.clip(0.0, 25.0))
    if mags:
        flow_mag = pd.concat(mags, axis=1).mean(axis=1, skipna=True).fillna(1.0)
    else:
        flow_mag = pd.Series(1.0, index=x.index)

    # Signed realised pressure: direction from tick movement, strength from
    # swap activity, penalised by available active liquidity.
    activity = np.sign(dlogp) * flow_mag / np.sqrt(liq_rel.clip(lower=0.05))
    activity = pd.Series(activity, index=x.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame(
        {
            "close_tick": close_tick,
            "activity": activity.clip(-25.0, 25.0),
        },
        index=x.index,
    )


def build_spatial_v8(
    q_idx: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    horizon_hours: int = 3,
    stat_window_hours: int = 72,
    fit_window_hours: int = 720,
    refit_every_hours: int = 24,
) -> pd.DataFrame:
    """Build the causal V8 spatial reduced-order detector.

    Output index is hourly. The detector uses only minute observations available
    through the end of each hour and fits dynamics only on completed historical
    hourly states.
    """
    required = {"close_tick", "current_liquidity"}
    if not required.issubset(q_idx.columns):
        return pd.DataFrame(index=hourly_index)

    minute = _minute_activity_proxy(q_idx)
    if minute.empty:
        return pd.DataFrame(index=hourly_index)

    # Fixed log-price coordinate relative to the first observed tick in this
    # experiment. Scale is fixed a priori (roughly +/- 150% in price).
    tick_ref = float(minute["close_tick"].dropna().iloc[0])
    xi = (minute["close_tick"] - tick_ref) * LOG_TICK
    z = (xi / 1.0).clip(-1.0, 1.0)
    psi = _basis(z.to_numpy())
    weighted = psi * minute["activity"].to_numpy()[:, None]

    coeff_minute = pd.DataFrame(weighted, index=minute.index, columns=[f"a{i}" for i in range(weighted.shape[1])])
    # Mean per minute keeps coefficient scale comparable across hours with a
    # small number of missing minute records.
    coeff_hour = coeff_minute.resample("1h").mean()
    coeff_hour = coeff_hour.reindex(hourly_index)

    n = coeff_hour.shape[1]
    out = pd.DataFrame(index=hourly_index)
    for col in coeff_hour.columns:
        out[f"v8_{col}"] = coeff_hour[col]

    score = np.full(len(out), np.nan)
    energy = np.full(len(out), np.nan)
    width = np.full(len(out), np.nan)
    intensity = np.full(len(out), np.nan)
    energy_f = np.full(len(out), np.nan)
    width_f = np.full(len(out), np.nan)
    intensity_f = np.full(len(out), np.nan)
    transfer = np.full(len(out), np.nan)
    diss = np.full(len(out), np.nan)
    lam = np.full(len(out), np.nan)
    fit_r2 = np.full(len(out), np.nan)

    model = None
    model_fit_at = -10**9
    arr = coeff_hour.to_numpy(dtype=float)

    for i in range(len(out)):
        if i < max(stat_window_hours, 96):
            continue
        hist_start = max(0, i - fit_window_hours)
        train = arr[hist_start : i + 1]
        train = train[np.isfinite(train).all(axis=1)]
        if len(train) < 96:
            continue

        if model is None or i - model_fit_at >= refit_every_hours:
            model = _fit_reduced_model(train)
            model_fit_at = i
        if model is None:
            continue

        state_hist = arr[max(0, i - stat_window_hours + 1) : i + 1]
        state_hist = state_hist[np.isfinite(state_hist).all(axis=1)]
        if len(state_hist) < 24:
            continue
        m = state_hist.mean(axis=0)
        c = np.cov(state_hist, rowvar=False)
        if c.ndim == 0:
            c = np.eye(n) * float(c)
        c = _psd(c)

        e0, l0, u0 = _energy_width(m, c)
        mf, cf = _forecast(model, m, c, horizon_hours)
        e1, l1, u1 = _energy_width(mf, cf)

        if e0 > 1e-12 and e1 > 1e-12 and l0 > 1e-12 and l1 > 1e-12:
            score[i] = 0.5 / horizon_hours * math.log((e1 / e0) * (l0 / l1))
        energy[i], width[i], intensity[i] = e0, l0, u0
        energy_f[i], width_f[i], intensity_f[i] = e1, l1, u1

        bbar = model.bbar_cov(c)
        pi = -float(m @ bbar)
        dloss = float(np.sum(model.damping * np.diag(c)))
        transfer[i] = pi
        diss[i] = dloss
        trc = float(np.trace(c))
        lam[i] = (pi - dloss) / trc if trc > 1e-12 else np.nan

        # Diagnostic in-sample fit quality on the current training block.
        pred = []
        real = []
        for j in range(len(train) - 1):
            pred.append(model.drift(train[j]))
            real.append(train[j + 1] - train[j])
        if pred:
            pp = np.asarray(pred)
            yy = np.asarray(real)
            sse = float(np.sum((yy - pp) ** 2))
            sst = float(np.sum((yy - yy.mean(axis=0)) ** 2))
            fit_r2[i] = 1.0 - sse / sst if sst > 1e-12 else np.nan

    out["v8_spatial_score"] = score
    out["v8_energy"] = energy
    out["v8_effective_width"] = width
    out["v8_local_intensity"] = intensity
    out["v8_forecast_energy"] = energy_f
    out["v8_forecast_width"] = width_f
    out["v8_forecast_local_intensity"] = intensity_f
    out["v8_transfer_pi"] = transfer
    out["v8_dissipation"] = diss
    out["v8_lambda"] = lam
    out["v8_fit_r2"] = fit_r2

    # Direct detector from the proposed mathematics.
    out["v8_concentration_state"] = (
        (out["v8_spatial_score"] > 0.0)
        & (out["v8_forecast_width"] < out["v8_effective_width"])
        & (out["v8_lambda"] > 0.0)
    )

    # A causal percentile is retained as a secondary severity diagnostic only.
    out["v8_score_threshold"] = (
        out["v8_spatial_score"]
        .rolling(168, min_periods=48)
        .quantile(0.90)
        .shift(1)
    )
    out["v8_severe_state"] = (
        out["v8_concentration_state"]
        & (out["v8_spatial_score"] > out["v8_score_threshold"])
    )
    return out
