"""
Synthetic regional wind-power verification experiment (companion to the copula paper).

The truth and three forecast systems are simulated at the real locations and nominal
capacities of the wind farms of the contiguous United States in the Global Power Plant
Database (GPPD v1.3.0, 1,256 unique sites, 113 GW).  Wind speed is mapped to power with
the NREL WIND Toolkit IEC Class II normalised power curve (as in Price et al. 2025).

Latent model for forecast case t (12-hourly cases, n of them):
    Z_t = sqrt(rho) * S_t + sqrt(1 - rho) * E_t
  S_t : predictable signal, Gaussian with exponential correlation (range ell_S), AR(1) in t
  E_t : unpredictable part with N(0,1) margins and a Student-t copula (nu, range ell_E)
  wind W_tj = Weibull_j^{-1}(Phi(Z_tj)),  power P_tj = cap_j * pc(W_tj)

Forecast systems (M members each, all know S_t):
  O : oracle      -- members drawn from the true conditional law
  A : calibrated margins, misspecified copula (Gaussian, range ell_A < ell_E; no tail dep.)
  B : true copula, mis-specified margins (smooth time-invariant offset field beta with
      s.d. tau, range ell_beta; spread factor kappa < 1)
Hybrids by copula transplantation (ECC-type reordering):
  AB : A's marginal values with B's ranks;  BA : B's values with A's ranks
  OA / OB : oracle marginals (ECC-R, common random numbers) with A's / B's ranks
  AP : A's values, independence copula (random permutation per site)
Placebo transplants (finite-ensemble leakage, Section 4.2 of the paper):
  PA : A's values with the ranks of a second, independent ensemble A' of system A
  PB : B's values with the ranks of a second, independent ensemble B' of system B
  A' and B' are drawn from a separate random stream (seed + 1), so switching the
  placebo on or off leaves every other number bit-for-bit unchanged.

Usage:
  python simulate_wind.py --out results/main.npz            (main experiment)
  python simulate_wind.py --out results/x.npz --n 365 --rhos 0.8 --sigma_beta 0.1 --ellA 300 --light
The RMS offset of system B is called sigma_beta in the paper (option --sigma_beta,
alias --tau, kept for backward compatibility with earlier file names).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

import copula_verif as cv
import icosahedral_mesh as im

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


# ------------------------------------------------------------------------------------
def load_sites():
    w = pd.read_csv(os.path.join(DATA, "gppd_wind.csv"))
    g = w.groupby(["latitude", "longitude"], as_index=False).agg(cap=("capacity_mw", "sum"))
    m = g.latitude.between(24, 50) & g.longitude.between(-125, -66)
    g = g[m].reset_index(drop=True)
    return g.longitude.values, g.latitude.values, g.cap.values


def power_curve():
    pc = pd.read_csv(os.path.join(DATA, "WTK_Validation_IEC-2_normalized.csv"))
    ws, p = pc.iloc[:, 0].values.astype(float), pc.iloc[:, 1].values.astype(float)
    return lambda W: np.interp(W, ws, p, left=0.0, right=0.0)


def build_pools(xyz_sites, radii_km, max_pools, rng):
    """Pool centres = nodes of the 7-times refined icosahedral mesh (as in Price et al.);
    keep centres with >= 1 site within radius r; subsample to at most max_pools."""
    mesh = im.get_last_triangular_mesh_for_sphere(splits=7)
    c = mesh.vertices.astype(np.float64)
    # restrict candidate centres to the neighbourhood of the sites (memory)
    cen = xyz_sites.mean(0)
    cen /= np.linalg.norm(cen)
    reach = np.arccos(np.clip(xyz_sites @ cen, -1, 1)).max() + max(radii_km) / cv.R_EARTH
    c = c[np.arccos(np.clip(c @ cen, -1, 1)) <= reach]
    pools = {}
    for r in radii_km:
        cos_r = np.cos(r / cv.R_EARTH)
        member = (c @ xyz_sites.T) >= cos_r                    # (n_centres, d)
        keep = np.where(member.sum(1) >= 1)[0]
        if len(keep) > max_pools:
            keep = np.sort(rng.choice(keep, max_pools, replace=False))
        lists = [np.where(member[k])[0] for k in keep]
        sizes = np.array([len(l) for l in lists])
        idx = np.concatenate(lists)
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        pools[r] = dict(idx=idx, starts=starts, sizes=sizes)
    return pools


def corr_exp(D, ell, nugget):
    return (1 - nugget) * np.exp(-D / ell) + nugget * np.eye(D.shape[0])


# ------------------------------------------------------------------------------------
def run(args):
    rng = np.random.default_rng(args.seed)
    rng2 = np.random.default_rng(args.seed + 1)     # independent stream: placebo ensembles only
    lon, lat, cap = load_sites()
    d = len(lon)
    xyz = cv.lonlat_to_xyz(lon, lat)
    D = cv.great_circle_matrix(xyz)
    pc = power_curve()
    # site-specific Weibull marginals (hub-height-like climatology)
    k_w = rng.uniform(1.8, 2.4, d)
    c_w = rng.uniform(7.0, 9.5, d)
    to_wind = lambda Z: c_w * (-np.log(norm.sf(Z))) ** (1.0 / k_w)

    L_S = np.linalg.cholesky(corr_exp(D, args.ellS, 1e-6))
    L_E = np.linalg.cholesky(corr_exp(D, args.ellE, args.nugget))
    L_A = np.linalg.cholesky(corr_exp(D, args.ellA, args.nugget))
    L_b = np.linalg.cholesky(corr_exp(D, args.ellbeta, 1e-6))
    beta = L_b @ rng.normal(size=d)                            # fixed offset field of system B
    beta *= args.tau / np.sqrt(np.mean(beta ** 2))             # RMS offset exactly tau
    nu = args.nu

    def meta_t(nrow, g=rng):
        G = g.normal(size=(nrow, d)) @ L_E.T
        V = g.chisquare(nu, size=(nrow, 1))
        return norm.ppf(np.clip(student_t.cdf(G / np.sqrt(V / nu), nu), 1e-12, 1 - 1e-12))

    def gauss_A(nrow, g=rng):
        return g.normal(size=(nrow, d)) @ L_A.T

    pools = build_pools(xyz, args.radii, args.max_pools, rng)
    radii = list(args.radii)
    M = args.M
    # pairs for the variogram score and for Kendall PITs
    iu = np.triu_indices(d, 1)
    Dp = D[iu]
    sel = np.where(Dp <= 300.0)[0]
    sel = rng.choice(sel, min(args.n_vs_pairs, len(sel)), replace=False)
    vs_pairs = np.c_[iu[0][sel], iu[1][sel]]
    near = np.where((Dp > 5) & (Dp <= 100))[0]
    mid = np.where((Dp > 200) & (Dp <= 500))[0]
    near = rng.choice(near, min(400, len(near)), replace=False)
    mid = rng.choice(mid, min(400, len(mid)), replace=False)
    pit_pairs = {"near": np.c_[iu[0][near], iu[1][near]], "mid": np.c_[iu[0][mid], iu[1][mid]]}
    pit_sites = rng.choice(d, 200, replace=False)

    ens_names = ["AA", "BB", "AB", "BA", "O", "AP", "OA", "OB"] + (["PA", "PB"] if args.placebo else [])
    score_names = (["site_crps", "vs", "es"] +
                   [f"{s}_{r}" for r in radii for s in ["avg", "max", "pow", "twpow", "brier"]])
    out = {}
    for rho in args.rhos:
        t0 = time.time()
        sr, se = np.sqrt(rho), np.sqrt(1 - rho)
        n = args.n
        S_prev = rng.normal(size=d) @ L_S.T
        scores = {e: {s: np.zeros(n) for s in score_names} for e in ens_names}
        pits = {e: {k: [] for k in pit_pairs} for e in ["AA", "BB", "O", "OA", "OB"]}
        mpit = {e: [] for e in ["AA", "BB"]}
        for t in range(n):
            S = args.phi * S_prev + np.sqrt(1 - args.phi ** 2) * (rng.normal(size=d) @ L_S.T)
            S_prev = S
            mu = sr * S
            E = meta_t(1 + 2 * M)                                # truth, O, B members
            z_obs = mu + se * E[0]
            zO = mu + se * E[1:M + 1]
            zB = mu + beta + args.kappa * se * E[M + 1:]
            zA = mu + se * gauss_A(M)
            obsW = to_wind(z_obs)
            W = {"O": to_wind(zO), "AA": to_wind(zA), "BB": to_wind(zB)}
            W["AB"] = cv.transplant(W["AA"], W["BB"], rng)
            W["BA"] = cv.transplant(W["BB"], W["AA"], rng)
            W["AP"] = cv.shuffle_independent(W["AA"], rng)
            rA, rB = cv.ranks(zA, rng), cv.ranks(zB, rng)
            # ECC-R on the oracle marginals with common random numbers: one sorted i.i.d.
            # sample per site, reordered by the ranks of A or of B (copula-only comparison)
            qs = np.sort(rng.normal(size=(M, d)), axis=0)
            W["OA"] = to_wind(mu + se * np.take_along_axis(qs, rA, axis=0))
            W["OB"] = to_wind(mu + se * np.take_along_axis(qs, rB, axis=0))
            if args.placebo:
                # second, independent ensembles of A and B (separate stream rng2); only their
                # ranks are used: PA = A's values + ranks of A', PB = B's values + ranks of B'
                zA2 = mu + se * gauss_A(M, rng2)
                zB2 = mu + beta + args.kappa * se * meta_t(M, rng2)
                W["PA"] = cv.transplant(W["AA"], zA2, rng2)
                W["PB"] = cv.transplant(W["BB"], zB2, rng2)
            obsP = cap * pc(obsW)
            for e in ens_names:
                X = W[e]
                P = cap[None, :] * pc(X)
                sc = scores[e]
                sc["site_crps"][t] = np.mean(cv.crps_fair(X, obsW))
                sc["vs"][t] = cv.variogram_score(X, obsW, vs_pairs, p=0.5) / len(vs_pairs)
                sc["es"][t] = cv.energy_score_fair(X, obsW)
                for r in radii:
                    pl = pools[r]
                    idx, st, sz = pl["idx"], pl["starts"], pl["sizes"]
                    xa = np.add.reduceat(X[:, idx], st, axis=1) / sz
                    ya = np.add.reduceat(obsW[idx], st) / sz
                    sc[f"avg_{r}"][t] = np.mean(cv.crps_fair(xa, ya))
                    xm = np.maximum.reduceat(X[:, idx], st, axis=1)
                    ym = np.maximum.reduceat(obsW[idx], st)
                    sc[f"max_{r}"][t] = np.mean(cv.crps_fair(xm, ym))
                    xp = np.add.reduceat(P[:, idx], st, axis=1)
                    yp = np.add.reduceat(obsP[idx], st)
                    capg = np.add.reduceat(cap[idx], st)
                    sc[f"pow_{r}"][t] = np.mean(cv.crps_fair(xp, yp))
                    thr = 0.10 * capg
                    sc[f"twpow_{r}"][t] = np.mean(cv.crps_fair(np.minimum(xp, thr), np.minimum(yp, thr)))
                    prob = np.mean(xp < 0.05 * capg, axis=0)
                    sc[f"brier_{r}"][t] = np.mean(cv.brier(prob, yp < 0.05 * capg))
            if not args.light:
                for e in pits:
                    for k, pr in pit_pairs.items():
                        pits[e][k].append(cv.coppit_pairs(W[e], obsW, pr, rng))
                for e in mpit:
                    mpit[e].append(cv.marginal_pit(W[e][:, pit_sites], obsW[pit_sites], rng))
        key = f"rho{rho}"
        for e in ens_names:
            for s in score_names:
                out[f"{key}/{e}/{s}"] = scores[e][s]
        if not args.light:
            for e in pits:
                for k in pit_pairs:
                    out[f"{key}/pit/{e}/{k}"] = np.concatenate(pits[e][k])
            for e in mpit:
                out[f"{key}/mpit/{e}"] = np.concatenate(mpit[e])
        print(f"rho={rho}: {n} cases in {time.time() - t0:.0f}s", flush=True)
    meta = dict(d=d, cap_total=cap.sum(), M=M, radii=np.array(radii),
                pool_counts=np.array([len(pools[r]["sizes"]) for r in radii]),
                pool_single=np.array([np.mean(pools[r]["sizes"] == 1) for r in radii]),
                pool_mean_cap=np.array([np.mean(np.add.reduceat(cap[pools[r]["idx"]], pools[r]["starts"])) for r in radii]),
                beta_sd=np.std(beta), n_vs_pairs=len(vs_pairs))
    for k, v in vars(args).items():
        if isinstance(v, (int, float)):
            meta[f"arg_{k}"] = v
    for k, v in meta.items():
        out[f"meta/{k}"] = np.asarray(v)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("saved", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "main.npz"))
    ap.add_argument("--n", type=int, default=730)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.8, 0.5, 0.2])
    ap.add_argument("--phi", type=float, default=0.7)
    ap.add_argument("--ellS", type=float, default=1000.0)
    ap.add_argument("--ellE", type=float, default=400.0)
    ap.add_argument("--ellA", type=float, default=150.0)
    ap.add_argument("--ellbeta", type=float, default=1500.0)
    ap.add_argument("--nugget", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=4.0)
    ap.add_argument("--sigma_beta", "--tau", dest="tau", type=float, default=0.2)
    ap.add_argument("--kappa", type=float, default=0.85)
    ap.add_argument("--radii", type=int, nargs="+", default=[60, 120, 240])
    ap.add_argument("--max_pools", type=int, default=600)
    ap.add_argument("--n_vs_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--light", action="store_true", help="skip PIT diagnostics")
    ap.add_argument("--no_placebo", dest="placebo", action="store_false",
                    help="skip the placebo transplants PA and PB")
    run(ap.parse_args())
