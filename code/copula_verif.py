"""
copula_verif.py -- copula-based tools for verifying ensemble forecasts of multivariate
(spatial / multi-site) quantities.  Companion code to

    "Separating marginal and dependence skill in machine-learning weather ensembles:
     a copula framework, with a simulation study for regional wind power"  (draft).

Everything works on plain numpy arrays.  Conventions for a single forecast case:

    ens : array (M, d)  -- M ensemble members, d components (sites, grid points, ...)
    obs : array (d,)    -- verifying values

Main entry points
-----------------
transplant(ens_marg, ens_cop, rng)   ensemble with the marginal values of `ens_marg`
                                     and the empirical copula (ranks) of `ens_cop`
shuffle_independent(ens, rng)        same marginals, independence copula
ecc_quantiles(ranks, ppf)            ECC-Q: attach given marginal quantile functions to ranks
crps_fair(ens, obs)                  fair (unbiased) ensemble CRPS, vectorised over columns
twcrps_fair(ens, obs, v)             threshold-weighted CRPS via a chaining function v
energy_score_fair(ens, obs)          fair energy score
variogram_score(ens, obs, pairs, p)  variogram score of order p on a list of pairs
coppit(ens, obs, rng)                ensemble Kendall (multivariate-rank) PIT, exactly uniform if ideal
shapley_2x2(s_AA, s_AB, s_BA, s_BB)  exact two-factor (marginals x copula) decomposition
block_bootstrap_ci(x, L, B, rng)     moving-block bootstrap CI for the mean of a series

Notation for the 2 x 2 design: s_XY = score of the forecast with the MARGINALS of
system X and the COPULA of system Y.  s_AA, s_BB are the raw systems; s_AB, s_BA are the
two transplanted hybrids.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------------------
# Copula manipulation for ensembles
# --------------------------------------------------------------------------------------

def ranks(ens: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Column-wise ranks 0..M-1 of an (M, d) ensemble; ties broken at random if rng given."""
    M = ens.shape[0]
    if rng is not None:
        jitter = rng.random(ens.shape)
        order = np.lexsort((jitter, ens), axis=0)
    else:
        order = np.argsort(ens, axis=0, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(M)[:, None].repeat(ens.shape[1], 1), axis=0)
    return r


def transplant(ens_marg: np.ndarray, ens_cop: np.ndarray,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Return the (M, d) ensemble whose j-th column contains exactly the values of
    ens_marg[:, j] (same multiset), re-ordered so that its ranks equal those of
    ens_cop[:, j].  Its empirical copula is therefore that of ens_cop (ECC / Schaake-type
    reordering).  Both ensembles must have the same size M."""
    if ens_marg.shape != ens_cop.shape:
        raise ValueError("ensembles must have the same shape (M, d)")
    sorted_vals = np.sort(ens_marg, axis=0)
    r = ranks(ens_cop, rng)
    return np.take_along_axis(sorted_vals, r, axis=0)


def shuffle_independent(ens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Destroy the dependence: independent random permutation of each column."""
    M, d = ens.shape
    perm = np.argsort(rng.random((M, d)), axis=0)
    return np.take_along_axis(ens, perm, axis=0)


def ecc_quantiles(r: np.ndarray, ppf) -> np.ndarray:
    """ECC-Q: given ranks r (M, d) in 0..M-1 and a vectorised quantile function
    ppf(u) -> (M, d) (or broadcastable), return members ppf((r + 1/2) / M)."""
    M = r.shape[0]
    return ppf((r + 0.5) / M)


# --------------------------------------------------------------------------------------
# Proper scores (fair / unbiased finite-ensemble versions where available)
# --------------------------------------------------------------------------------------

def crps_fair(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS (Ferro 2014) for each column of an (M, k) ensemble; returns (k,)."""
    M = ens.shape[0]
    xs = np.sort(ens, axis=0)
    t1 = np.mean(np.abs(ens - obs[None, :]), axis=0)
    coef = (2 * np.arange(1, M + 1) - M - 1)[:, None]
    t2 = np.sum(coef * xs, axis=0) / (M * (M - 1))
    return t1 - t2


def twcrps_fair(ens: np.ndarray, obs: np.ndarray, v) -> np.ndarray:
    """Threshold-weighted CRPS through a chaining function v (Allen et al. 2023):
    twCRPS(F, y) = CRPS(F o v^{-1}, v(y)).  e.g. lower tail below r: v = lambda z: np.minimum(z, r)."""
    return crps_fair(v(ens), v(obs))


def energy_score_fair(ens: np.ndarray, obs: np.ndarray) -> float:
    """Fair energy score (beta = 1) of an (M, d) ensemble."""
    M = ens.shape[0]
    t1 = np.mean(np.linalg.norm(ens - obs[None, :], axis=1))
    G = ens @ ens.T
    sq = np.diag(G)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0)
    t2 = np.sqrt(D2).sum() / (2 * M * (M - 1))
    return t1 - t2


def variogram_score(ens: np.ndarray, obs: np.ndarray, pairs: np.ndarray,
                    p: float = 0.5, w: np.ndarray | None = None) -> float:
    """Variogram score of order p (Scheuerer & Hamill 2015) over index pairs (n, 2)."""
    i, j = pairs[:, 0], pairs[:, 1]
    vo = np.abs(obs[i] - obs[j]) ** p
    vf = np.mean(np.abs(ens[:, i] - ens[:, j]) ** p, axis=0)
    w = np.ones_like(vo) if w is None else w
    return float(np.sum(w * (vo - vf) ** 2))


def brier(prob: np.ndarray, event: np.ndarray) -> np.ndarray:
    return (prob - event.astype(float)) ** 2


# --------------------------------------------------------------------------------------
# Copula calibration: Kendall / multivariate-rank PIT (Gneiting et al. 2008;
# Ziegel & Gneiting 2014).  The pre-rank of each point of the pooled set
# {y, x_1, ..., x_M} is its empirical joint cdf value within the pooled set,
# rho_j = #{i : s_i <= s_j componentwise}, i.e. an empirical version of H(.) whose
# distribution function is the Kendall function K_H.  The randomised rank of rho_0 is
# exactly uniform when y and the members are exchangeable (ideal forecast), and as
# M -> infinity it converges to the copula PIT K_H(H(Y)) of Ziegel & Gneiting (2014).
# --------------------------------------------------------------------------------------

def _rank_pit(pre: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """pre: (M+1, n) pre-ranks with row 0 = observation.  Returns randomised PIT (n,)."""
    M1 = pre.shape[0]
    below = (pre[1:] < pre[0][None, :]).sum(axis=0)
    ties = (pre[1:] == pre[0][None, :]).sum(axis=0)
    r = below + np.floor(rng.random(below.shape) * (ties + 1))   # 0..M, uniform tie-break
    return (r + rng.random(below.shape)) / M1


def coppit(ens: np.ndarray, obs: np.ndarray, rng: np.random.Generator) -> float:
    """Kendall (multivariate-rank) PIT of a d-variate observation w.r.t. an (M, d) ensemble."""
    S = np.vstack([obs[None, :], ens])                          # (M+1, d)
    pre = np.all(S[:, None, :] <= S[None, :, :], axis=2).sum(axis=0)   # (M+1,)
    return float(_rank_pit(pre[:, None], rng)[0])


def coppit_pairs(ens: np.ndarray, obs: np.ndarray, pairs: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """Vectorised bivariate Kendall PIT for many index pairs (n, 2); returns (n,)."""
    a = np.vstack([obs[None, pairs[:, 0]], ens[:, pairs[:, 0]]])   # (M+1, n)
    b = np.vstack([obs[None, pairs[:, 1]], ens[:, pairs[:, 1]]])
    le = (a[:, None, :] <= a[None, :, :]) & (b[:, None, :] <= b[None, :, :])   # (M+1, M+1, n)
    pre = le.sum(axis=0)                                         # (M+1, n)
    return _rank_pit(pre, rng)


def marginal_pit(ens: np.ndarray, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomised univariate rank PIT for each column; returns (d,)."""
    pre = np.vstack([obs[None, :], ens])
    return _rank_pit(pre, rng)


# --------------------------------------------------------------------------------------
# Decomposition and inference
# --------------------------------------------------------------------------------------

def shapley_2x2(s_AA, s_AB, s_BA, s_BB):
    """Exact two-factor Shapley decomposition of s_AA - s_BB into a marginal and a copula
    part.  s_XY = score with marginals of X and copula of Y.  Works element-wise, so it can
    be applied to per-case scores (the decomposition is linear in the scores).
    Returns dict(total, marg, cop, inter) with total = marg + cop exactly and
    inter = (s_AA - s_AB) - (s_BA - s_BB) (copula effect under A- vs B-marginals)."""
    total = s_AA - s_BB
    cop = 0.5 * ((s_AA - s_AB) + (s_BA - s_BB))
    marg = 0.5 * ((s_AA - s_BA) + (s_AB - s_BB))
    inter = (s_AA - s_AB) - (s_BA - s_BB)
    return dict(total=total, marg=marg, cop=cop, inter=inter)


def block_bootstrap_ci(x: np.ndarray, L: int = 10, B: int = 2000, level: float = 0.95,
                       rng: np.random.Generator | None = None):
    """Moving-block bootstrap percentile CI for the mean of a (possibly autocorrelated)
    series x (n,) or several series (n, k) at once.  Returns (mean, lo, hi)."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n]
    means = x[idx].mean(axis=1)                                  # (B, k)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a], axis=0)
    return x.mean(axis=0), lo, hi


def benjamini_yekutieli(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Yekutieli FDR control under arbitrary dependence; returns reject mask."""
    p = np.asarray(pvals).ravel()
    m = p.size
    cm = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / (m * cm)
    ok = p[order] <= thresh
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej.reshape(np.shape(pvals))


def block_bootstrap_pvalue(x: np.ndarray, L: int = 10, B: int = 2000,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """Two-sided bootstrap p-value for H0: E[x] = 0 (moving blocks, centred resampling)."""
    rng = np.random.default_rng(1) if rng is None else rng
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n]
    xc = x - x.mean(axis=0)
    t_obs = np.abs(x.mean(axis=0))
    t_b = np.abs(xc[idx].mean(axis=1))
    return (1 + np.sum(t_b >= t_obs[None, :], axis=0)) / (B + 1)


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
R_EARTH = 6371.0


def lonlat_to_xyz(lon_deg, lat_deg):
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]


def great_circle_matrix(xyz: np.ndarray) -> np.ndarray:
    c = np.clip(xyz @ xyz.T, -1.0, 1.0)
    return np.arccos(c) * R_EARTH
