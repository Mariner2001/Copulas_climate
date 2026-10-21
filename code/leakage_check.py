"""Finite-ensemble check of copula transplantation (Section 4.2 and Appendix B of the paper).

Bivariate truth with N(0,1) margins and a t_4 copula (correlation parameter 0.8).
System A: correct margins, Gaussian copula with the same correlation parameter.
System B: the truth.  Because the margins coincide, H_AB = H_B and H_BA = H_A, so the
leakage of each hybrid can be measured directly with paired differences:
    L_AB = E S(T(a, b)) - E S(b),     L_BA = E S(T(b, a)) - E S(a),
and compared with the placebo estimates obtained from second, independent ensembles
a' and b' of the same systems:
    Lpl_A = E S(T(a, a')) - E S(a),   Lpl_B = E S(T(b, b')) - E S(b).
Placebo ensembles come from a separate random stream, so the other columns are identical
with and without them.  Scores: variogram score (p = 0.5) and fair CRPS of the sum.
"""
import os
import numpy as np
from scipy.stats import norm, t as tdist
import copula_verif as cv

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "..", "tables")
RES = os.path.join(HERE, "..", "results")
rng = np.random.default_rng(5)
rng2 = np.random.default_rng(6)          # placebo stream
nu, r = 4, 0.8
L = np.linalg.cholesky(np.array([[1, r], [r, 1]]))


def meta_t(k, g):
    G = g.normal(size=(k, 2)) @ L.T
    V = g.chisquare(nu, size=(k, 1))
    return norm.ppf(tdist.cdf(G / np.sqrt(V / nu), nu))


def gauss(k, g):
    return g.normal(size=(k, 2)) @ L.T


pairs = np.array([[0, 1]])
COLS = ["raw", "copA", "copB", "shap", "LAB", "LplA", "LBA", "LplB"]
rows = []
for M, n in [(20, 40000), (50, 40000), (200, 20000)]:
    S = {k: {"vs": [], "sum": []} for k in ["AA", "BB", "AB", "BA", "PA", "PB"]}
    for _ in range(n):
        y = meta_t(1, rng)[0]
        A, B = gauss(M, rng), meta_t(M, rng)
        A2, B2 = gauss(M, rng2), meta_t(M, rng2)
        E = {"AA": A, "BB": B, "AB": cv.transplant(A, B), "BA": cv.transplant(B, A),
             "PA": cv.transplant(A, A2), "PB": cv.transplant(B, B2)}
        for k, X in E.items():
            S[k]["vs"].append(cv.variogram_score(X, y, pairs, p=0.5))
            S[k]["sum"].append(cv.crps_fair(X.sum(1, keepdims=True), np.array([y.sum()]))[0])
    for sc in ["vs", "sum"]:
        a = {k: np.array(S[k][sc]) for k in S}
        base = a["BB"].mean()
        terms = {
            "raw": a["AA"] - a["BB"],
            "copA": a["AA"] - a["AB"],
            "copB": a["BA"] - a["BB"],
            "shap": 0.5 * ((a["AA"] - a["AB"]) + (a["BA"] - a["BB"])),
            "LAB": a["AB"] - a["BB"],
            "LplA": a["PA"] - a["AA"],
            "LBA": a["BA"] - a["AA"],
            "LplB": a["PB"] - a["BB"],
        }
        vals = [(100 * terms[c].mean() / base, 100 * terms[c].std(ddof=1) / np.sqrt(n) / base) for c in COLS]
        rows.append((sc, M, vals))
        print(sc, M, " ".join(f"{c}={m:+.2f}({e:.2f})" for c, (m, e) in zip(COLS, vals)), flush=True)

os.makedirs(RES, exist_ok=True)
with open(os.path.join(RES, "leakage_check.txt"), "w") as fh:
    for sc, M, vals in rows:
        fh.write(f"{sc} M={M} " + " ".join(f"{c}={m:+.3f}({e:.3f})" for c, (m, e) in zip(COLS, vals)) + "\n")

lab = {"vs": "Variogram score ($p=0.5$)", "sum": "CRPS of the sum"}


def cell(m, e):
    return f"{m:+.2f} ({e:.2f})".replace("-", "$-$")


lines = []
for sc in ["vs", "sum"]:
    for (s, M, vals) in rows:
        if s != sc:
            continue
        lines.append(f"{lab[sc] if M == 20 else ''} & {M} & " + " & ".join(cell(m, e) for m, e in vals) + r" \\")
    lines.append(r"\addlinespace")
tex = r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{Finite-ensemble behaviour of the transplant. Bivariate truth with $N(0,1)$ marginals and a $t_4$ copula (correlation parameter 0.8); system A has the same marginals and a Gaussian copula with the same correlation parameter; system B is the truth. Entries are in \% of B's mean score, with Monte Carlo standard errors in brackets ($n=40{,}000$ cases for $M=20,50$ and $20{,}000$ for $M=200$). Because the marginals coincide, the raw difference is a pure copula effect, and the leakages can be measured directly: $L_{AB}=\bar s_{AB}-\bar s_{BB}$ and $L_{BA}=\bar s_{BA}-\bar s_{AA}$. The placebo estimates $\hat L_A$ and $\hat L_B$ use second, independent ensembles of A and B.}\label{tab:leak}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l r rrrr rrrr@{}}
\toprule
 & & \multicolumn{4}{c}{Copula effects} & \multicolumn{4}{c}{Leakage}\\
\cmidrule(lr){3-6}\cmidrule(l){7-10}
Score & $M$ & raw $\bar s_{AA}-\bar s_{BB}$ & $\bar s_{AA}-\bar s_{AB}$ & $\bar s_{BA}-\bar s_{BB}$ & Shapley $\hat\Delta_{\mathrm{cop}}$ & $L_{AB}$ & placebo $\hat L_A$ & $L_{BA}$ & placebo $\hat L_B$\\
\midrule
BODY
\bottomrule
\end{tabular}}
\end{table}
""".replace("BODY", "\n".join(lines[:-1]))
os.makedirs(TAB, exist_ok=True)
open(os.path.join(TAB, "tab_leakage.tex"), "w").write(tex)
print("written", os.path.join(TAB, "tab_leakage.tex"))
