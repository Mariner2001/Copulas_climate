"""Figures and tables for the copula paper (reads results/*.npz written by simulate_wind.py).

Column order of the per-score arrays written to results/main_decomposition.json
(all in % of B's mean score; mean, lower and upper 95% moving-block bootstrap limits):
  0 total A-B, 1 Shapley marginal effect, 2 Shapley copula effect,
  3 AA-AB (one-sided copula effect, A's marginals), 4 BA-BB (one-sided, B's marginals),
  5 OA-OB (copula-only comparison on oracle marginals, ECC-R),
  6 AP-AA (independence copula vs A's copula), 7 AA-O, 8 BB-O,
  9 PA-AA (placebo leakage, recipient A), 10 PB-BB (placebo leakage, recipient B),
  11 PA-AB (placebo-matched copula effect, A's marginals),
  12 BA-PB (placebo-matched copula effect, B's marginals), 13 mean of 11 and 12.
p-values (two-sided, centred moving-block bootstrap) are stored for columns 0-2 only.
"""
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import copula_verif as cv

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
C7 = "#6250d6"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False, "pdf.fonttype": 42,
})

LABELS = {
    "site_crps": "site CRPS (wind)", "avg_60": "avg-pooled CRPS, r=60 km", "avg_240": "avg-pooled CRPS, r=240 km",
    "max_60": "max-pooled CRPS, r=60 km", "max_240": "max-pooled CRPS, r=240 km",
    "pow_60": "regional power CRPS, r=60 km", "pow_120": "regional power CRPS, r=120 km",
    "pow_240": "regional power CRPS, r=240 km", "twpow_120": "low-power twCRPS, r=120 km",
    "brier_120": "wind-drought Brier, r=120 km", "vs": "variogram score (p=0.5)", "es": "energy score",
}
ORDER = ["site_crps", "avg_60", "avg_240", "max_60", "max_240", "pow_60", "pow_120", "pow_240",
         "twpow_120", "brier_120", "vs", "es"]


# --------------------------------------------------------------------------------------
def fig_schematic():
    """Copula transplantation for two neighbouring sites (synthetic, latent scale)."""
    rng = np.random.default_rng(3)
    M = 50
    # system A: Gaussian copula, weak dependence (r=0.3), calibrated N(0,1) margins
    LA = np.linalg.cholesky(np.array([[1, .3], [.3, 1]]))
    A = rng.normal(size=(M, 2)) @ LA.T
    # system B: t4 copula, strong dependence (r=0.85), offset and under-dispersed margins
    LB = np.linalg.cholesky(np.array([[1, .85], [.85, 1]]))
    G = rng.normal(size=(M, 2)) @ LB.T
    T = G / np.sqrt(rng.chisquare(4, size=(M, 1)) / 4)
    B = 0.6 + 0.7 * norm.ppf(student_t.cdf(T, 4))
    AB = cv.transplant(A, B)
    BA = cv.transplant(B, A)
    panels = [(A, "A: A-marginals, A-copula", C1), (AB, "AB: A-marginals, B-copula", C1),
              (BA, "BA: B-marginals, A-copula", C2), (B, "B: B-marginals, B-copula", C2)]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), constrained_layout=True, sharex=True, sharey=True)
    for ax, (X, title, col) in zip(axs, panels):
        ax.scatter(X[:, 0], X[:, 1], s=10, color=col, edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot(X[:, 0], np.full(M, -3.3), "|", color=col, ms=4, alpha=0.6)
        ax.plot(np.full(M, -3.3), X[:, 1], "_", color=col, ms=4, alpha=0.6)
        tau = np.corrcoef(cv.ranks(X)[:, 0], cv.ranks(X)[:, 1])[0, 1]
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.text(0.14, 0.95, f"rank corr. {tau:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK2)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel("site 1")
    axs[0].set_ylabel("site 2")
    fig.savefig(os.path.join(FIG, "fig1_transplant_schematic.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------------------
def decomposition_table(path, B=2000, L=10):
    z = np.load(path)
    rhos = sorted({k.split("/")[0] for k in z.files if k.startswith("rho")}, reverse=True)
    out = {}
    for rk in rhos:
        out[rk] = {}
        for s in ORDER:
            g = lambda e: z[f"{rk}/{e}/{s}"]
            sh = cv.shapley_2x2(g("AA"), g("AB"), g("BA"), g("BB"))
            base = g("BB").mean()
            cols = [sh["total"], sh["marg"], sh["cop"], g("AA") - g("AB"), g("BA") - g("BB"),
                    g("OA") - g("OB"), g("AP") - g("AA"), g("AA") - g("O"), g("BB") - g("O")]
            if f"{rk}/PA/{s}" in z.files:
                # placebo leakage estimates and placebo-matched conditional copula effects
                mA, mB = g("PA") - g("AB"), g("BA") - g("PB")
                cols += [g("PA") - g("AA"), g("PB") - g("BB"), mA, mB, 0.5 * (mA + mB)]
            X = np.column_stack(cols)
            m, lo, hi = cv.block_bootstrap_ci(X, L=L, B=B, rng=np.random.default_rng(11))
            p = cv.block_bootstrap_pvalue(X[:, :3], L=L, B=B, rng=np.random.default_rng(12))
            out[rk][s] = dict(mean=(100 * m / base).tolist(), lo=(100 * lo / base).tolist(),
                              hi=(100 * hi / base).tolist(), p=p.tolist(), base=float(base))
    return out


def fig_decomposition(tab, fname="fig2_decomposition.pdf"):
    rks = list(tab.keys())
    titles = {"rho0.8": r"short lead ($\rho$ = 0.8)", "rho0.5": r"medium lead ($\rho$ = 0.5)",
              "rho0.2": r"long lead ($\rho$ = 0.2)"}
    fig, axs = plt.subplots(1, len(rks), figsize=(7.2, 4.1), sharey=True, constrained_layout=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, rk in zip(np.atleast_1d(axs), rks):
        mm = np.array([tab[rk][s]["mean"] for s in ORDER])
        lo = np.array([tab[rk][s]["lo"] for s in ORDER])
        hi = np.array([tab[rk][s]["hi"] for s in ORDER])
        ax.barh(y + 0.18, mm[:, 1], height=0.34, color=C3, label="marginal effect")
        ax.barh(y - 0.18, mm[:, 2], height=0.34, color=C7, label="copula effect")
        ax.errorbar(mm[:, 0], y, xerr=[mm[:, 0] - lo[:, 0], hi[:, 0] - mm[:, 0]], fmt="D", ms=3.5,
                    color=INK, ecolor=INK, elinewidth=0.9, capsize=1.5, label="total (95% CI)")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(titles.get(rk, rk), loc="left")
        ax.set_xlabel("A minus B, % of B's score")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([LABELS[s] for s in ORDER])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.6, -0.06))
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)


def fig_pit(path, rk="rho0.8", fname="fig3_kendall_pit.pdf"):
    z = np.load(path)
    cases = [("O", "oracle"), ("OB", "oracle margins + B copula"), ("OA", "oracle margins + A copula"),
             ("BB", "raw B (offset margins)")]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.3), sharex=True, constrained_layout=True)
    bins = np.linspace(0, 1, 11)
    for i, (k, kl) in enumerate([("near", "pairs < 100 km"), ("mid", "pairs 200-500 km")]):
        for j, (e, lab) in enumerate(cases):
            u = z[f"{rk}/pit/{e}/{k}"]
            h, _ = np.histogram(u, bins=bins)
            dens = h / h.sum() * (len(bins) - 1)
            col = {"O": INK2, "OB": C2, "OA": C1, "BB": C2}[e]
            axs[i, j].bar(bins[:-1], dens, width=0.098, align="edge", color=col,
                          alpha=0.55 if e == "BB" else 1.0)
            axs[i, j].axhline(1, color=INK, lw=0.8, ls="--")
            axs[i, j].set_ylim(0, 2.0)
            axs[i, j].grid(axis="x", visible=False)
            if i == 0:
                axs[i, j].set_title(lab, loc="left", fontsize=7.2)
            if j == 0:
                axs[i, j].set_ylabel(f"{kl}\ndensity")
            if i == 1:
                axs[i, j].set_xlabel("Kendall PIT")
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def pit_stats(path, rk):
    """Simple uniformity summaries of Kendall PITs: KS statistic and share in outer deciles."""
    from scipy.stats import kstest
    z = np.load(path)
    out = {}
    for e in ["O", "OA", "OB", "AA", "BB"]:
        for k in ["near", "mid"]:
            u = z[f"{rk}/pit/{e}/{k}"]
            out[f"{e}/{k}"] = dict(ks=float(kstest(u, "uniform").statistic),
                                   outer=float(np.mean((u < 0.1) | (u > 0.9))), n=int(u.size))
    for e in ["AA", "BB"]:
        u = z[f"{rk}/mpit/{e}"]
        out[f"mpit/{e}"] = dict(ks=float(kstest(u, "uniform").statistic),
                                outer=float(np.mean((u < 0.1) | (u > 0.9))), mean=float(u.mean()))
    return out


def sweep_table(score="pow_240"):
    files = sorted(glob.glob(os.path.join(RES, "sweep_tau*_ellA*.npz")))
    rows = []
    for f in files:
        b = os.path.basename(f)[:-4]
        tau = float(b.split("_")[1][3:]); ellA = float(b.split("_")[2][4:])
        z = np.load(f)
        rk = [k for k in z.files if k.startswith("rho")][0].split("/")[0]
        g = lambda e: z[f"{rk}/{e}/{score}"]
        sh = cv.shapley_2x2(g("AA"), g("AB"), g("BA"), g("BB"))
        base = g("BB").mean()
        X = np.c_[sh["total"], sh["cop"], sh["marg"], g("AA") - g("AB"), g("BA") - g("BB"), g("OA") - g("OB")]
        m, lo, hi = cv.block_bootstrap_ci(X, L=10, B=1000, rng=np.random.default_rng(5))
        row = dict(tau=tau, ellA=ellA)
        for i, k in enumerate(["total", "cop", "marg", "copA", "copB", "oaob"]):
            row[k] = 100 * m[i] / base
            row[k + "_lo"] = 100 * lo[i] / base
            row[k + "_hi"] = 100 * hi[i] / base
        rows.append(row)
    return rows


def fig_sensitivity(fname="fig4_sensitivity.pdf", scores=("pow_240", "vs"), extra=("max_240", "twpow_120", "brier_120")):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    titles = {"pow_240": "regional power CRPS, r = 240 km", "vs": "variogram score (p = 0.5)"}
    fig, axs = plt.subplots(len(scores), 2, figsize=(7.2, 2.35 * len(scores)), constrained_layout=True)
    axs = np.atleast_2d(axs)
    allrows = {}
    for i, sc in enumerate(scores):
        rows = sweep_table(sc)
        allrows[sc] = rows
        taus = sorted({r["tau"] for r in rows}); ells = sorted({r["ellA"] for r in rows})
        for j, key in enumerate(["total", "cop"]):
            Z = np.full((len(taus), len(ells)), np.nan)
            S = np.zeros_like(Z, dtype=bool)
            for r in rows:
                a, b = taus.index(r["tau"]), ells.index(r["ellA"])
                Z[a, b] = r[key]
                lo, hi = r[key + "_lo"], r[key + "_hi"]
                S[a, b] = (lo > 0) or (hi < 0)
            lim = np.nanmax(np.abs(Z))
            ax = axs[i, j]
            # negative = A better (blue), positive = B better (red)
            im = ax.imshow(-Z, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), origin="lower", aspect="auto")
            for a in range(len(taus)):
                for b in range(len(ells)):
                    txt = f"{Z[a, b]:+.1f}" + ("" if S[a, b] else "\u2009ns")
                    ax.text(b, a, txt, ha="center", va="center", fontsize=7, color=INK)
            ax.set_xticks(range(len(ells))); ax.set_xticklabels([f"{int(e)}" for e in ells])
            ax.set_yticks(range(len(taus))); ax.set_yticklabels([f"{t:.1f}" for t in taus])
            ax.set_xlabel("copula range of system A (km; truth 400)")
            ax.set_ylabel("RMS offset of system B")
            ax.grid(False)
            lab = "total A minus B (%)" if key == "total" else "copula effect (%)"
            ax.set_title(f"{titles[sc]}: {lab}", loc="left", fontsize=7.6)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    for sc in extra:
        allrows[sc] = sweep_table(sc)
    return allrows


if __name__ == "__main__":
    fig_schematic()
    main = os.path.join(RES, "main.npz")
    if os.path.exists(main):
        tab = decomposition_table(main)
        json.dump(tab, open(os.path.join(RES, "main_decomposition.json"), "w"), indent=1)
        fig_decomposition(tab)
        fig_pit(main, "rho0.8")
        fig_pit(main, "rho0.2", "figS1_kendall_pit_long.pdf")
        stats = {rk: pit_stats(main, rk) for rk in ["rho0.8", "rho0.5", "rho0.2"]}
        json.dump(stats, open(os.path.join(RES, "main_pit_stats.json"), "w"), indent=1)
        z = np.load(main)
        meta = {k.split("/", 1)[1]: z[k].tolist() for k in z.files if k.startswith("meta/")}
        json.dump(meta, open(os.path.join(RES, "main_meta.json"), "w"), indent=1)
    if glob.glob(os.path.join(RES, "sweep_tau*_ellA*.npz")):
        rows = fig_sensitivity()
        json.dump(rows, open(os.path.join(RES, "sweep_table.json"), "w"), indent=1)
    print("figures written to", FIG)
