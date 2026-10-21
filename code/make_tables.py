"""LaTeX tables for the copula paper from results/main_decomposition.json.

tab_main.tex     Table of Section 7.2: total, Shapley marginal and copula effects, OA-OB.
                 Significance: Benjamini-Yekutieli FDR control at 5% over the 105 tests of
                 total, marginal and copula terms that are not identically zero (the copula
                 terms of the site CRPS vanish identically, Proposition 2a, and are excluded).
tab_placebo.tex  Table of Appendix B: placebo leakage and placebo-matched copula effects.
"""
import os
import json
import numpy as np
import copula_verif as cv

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(TAB, exist_ok=True)
LAB = {"site_crps": "Site CRPS (wind speed)", "avg_60": "Avg-pooled CRPS, $r=60$\\,km",
       "avg_240": "Avg-pooled CRPS, $r=240$\\,km", "max_60": "Max-pooled CRPS, $r=60$\\,km",
       "max_240": "Max-pooled CRPS, $r=240$\\,km", "pow_60": "Regional power CRPS, $r=60$\\,km",
       "pow_120": "Regional power CRPS, $r=120$\\,km", "pow_240": "Regional power CRPS, $r=240$\\,km",
       "twpow_120": "Low-power twCRPS, $r=120$\\,km", "brier_120": "Wind-drought Brier, $r=120$\\,km",
       "vs": "Variogram score ($p=0.5$)", "es": "Energy score"}
ORDER = list(LAB)
RKS = ["rho0.8", "rho0.5", "rho0.2"]
tab = json.load(open(os.path.join(RES, "main_decomposition.json")))


def fmt(v, nd=1):
    if abs(v) < 0.5 * 10 ** (-nd):
        return "0." + "0" * nd
    return f"{v:+.{nd}f}".replace("-", "$-$")


def cell(v, sig):
    txt = fmt(v)
    return txt if sig else f"\\textcolor{{gray}}{{{txt}}}"


# ---- multiplicity: BY over the 105 non-degenerate tests -------------------------------
keys, pvals = [], []
for rk in RKS:
    for s in ORDER:
        for i in range(3):
            if s == "site_crps" and i == 2:
                continue                       # identically zero copula term
            keys.append((rk, s, i))
            pvals.append(tab[rk][s]["p"][i])
rej_flat = cv.benjamini_yekutieli(np.array(pvals), q=0.05)
REJ = {k: bool(r) for k, r in zip(keys, rej_flat)}
N_TESTS = len(keys)
json.dump({"n_tests": N_TESTS, "n_rejected": int(rej_flat.sum()),
           "not_rejected": [list(k) for k, r in REJ.items() if not r]},
          open(os.path.join(RES, "by_fdr.json"), "w"), indent=1)

# ---- main table ---------------------------------------------------------------------
rows = []
for s in ORDER:
    vals = []
    for rk in RKS:
        m = tab[rk][s]["mean"]
        if s == "site_crps":                   # copula terms vanish identically
            vals.append([cell(m[0], REJ[(rk, s, 0)]), cell(m[1], REJ[(rk, s, 1)]), "0", "0"])
        else:
            vals.append([cell(m[0], REJ[(rk, s, 0)]), cell(m[1], REJ[(rk, s, 1)]),
                         cell(m[2], REJ[(rk, s, 2)]), fmt(m[5])])
    rows.append(LAB[s] + " & " + " && ".join(" & ".join(v) for v in vals) + r" \\")
tex = r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{3.2pt}
\caption{Simulation results: score of system A minus score of system B, in \% of B's mean score (negative values favour A), with the Shapley split into marginal and copula effects and the copula-only comparison OA$-$OB on common oracle marginals (ECC-R). A has calibrated marginals and a Gaussian copula of range 150\,km; B has the true $t_4$ copula (range 400\,km) and offset, under-dispersed marginals. 730 cases per lead time, 50 members. Grey entries are not significant after Benjamini--Yekutieli control at 5\% over the NTESTS tests of total, marginal and copula effects (moving-block bootstrap, blocks of 10 cases); the copula effects of the site CRPS are identically zero and the OA$-$OB column is not tested.}\label{tab:main}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l rrrr c rrrr c rrrr@{}}
\toprule
 & \multicolumn{4}{c}{Short lead ($\rho=0.8$)} && \multicolumn{4}{c}{Medium lead ($\rho=0.5$)} && \multicolumn{4}{c}{Long lead ($\rho=0.2$)}\\
\cmidrule{2-5}\cmidrule{7-10}\cmidrule{12-15}
Score & total & marg. & cop. & OA$-$OB && total & marg. & cop. & OA$-$OB && total & marg. & cop. & OA$-$OB\\
\midrule
BODY
\bottomrule
\end{tabular}}
\end{table}
""".replace("BODY", "\n".join(rows)).replace("NTESTS", str(N_TESTS))
open(os.path.join(TAB, "tab_main.tex"), "w").write(tex)
print(tex)

# ---- placebo table (Appendix B) -------------------------------------------------------
if len(tab[RKS[0]][ORDER[0]]["mean"]) > 13:
    rows = []
    for s in ORDER:
        vals = []
        for rk in RKS:
            m = tab[rk][s]["mean"]
            vals.append([fmt(m[9], 2), fmt(m[10], 2), fmt(m[2], 2), fmt(m[13], 2)])
        rows.append(LAB[s] + " & " + " && ".join(" & ".join(v) for v in vals) + r" \\")
    tex = r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{Placebo transplants in the main experiment, in \% of B's mean score: leakage estimates $\hat L_A$ (recipient A) and $\hat L_B$ (recipient B), and the Shapley copula effect $\hat\Delta_{\mathrm{cop}}$ of Table~\ref{tab:main} compared with its placebo-matched version $\hat\Delta^{\mathrm{m}}_{\mathrm{cop}}$ (Definition~\ref{def:matched}). Positive copula effects favour B.}\label{tab:placebo}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l rrrr c rrrr c rrrr@{}}
\toprule
 & \multicolumn{4}{c}{Short lead ($\rho=0.8$)} && \multicolumn{4}{c}{Medium lead ($\rho=0.5$)} && \multicolumn{4}{c}{Long lead ($\rho=0.2$)}\\
\cmidrule{2-5}\cmidrule{7-10}\cmidrule{12-15}
Score & $\hat L_A$ & $\hat L_B$ & $\hat\Delta_{\mathrm{cop}}$ & $\hat\Delta^{\mathrm{m}}_{\mathrm{cop}}$ && $\hat L_A$ & $\hat L_B$ & $\hat\Delta_{\mathrm{cop}}$ & $\hat\Delta^{\mathrm{m}}_{\mathrm{cop}}$ && $\hat L_A$ & $\hat L_B$ & $\hat\Delta_{\mathrm{cop}}$ & $\hat\Delta^{\mathrm{m}}_{\mathrm{cop}}$\\
\midrule
BODY
\bottomrule
\end{tabular}}
\end{table}
""".replace("BODY", "\n".join(rows))
    open(os.path.join(TAB, "tab_placebo.tex"), "w").write(tex)
    print(tex)
