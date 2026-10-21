"""Write the Results text of Section 7.2 (tables/results_text.tex) and the placebo paragraphs
of Appendix B (tables/placebo_text.tex).  Every number is read from the JSON outputs of
make_figures.py, make_tables.py and leakage_check.py, so the text cannot drift from the
results.  Column indices of the decomposition arrays are documented in make_figures.py.
"""
import os
import json
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "tables")
tab = json.load(open(os.path.join(RES, "main_decomposition.json")))
pit = json.load(open(os.path.join(RES, "main_pit_stats.json")))
by = json.load(open(os.path.join(RES, "by_fdr.json")))
sweep_path = os.path.join(RES, "sweep_table.json")
sweep = json.load(open(sweep_path)) if os.path.exists(sweep_path) else None
leak_path = os.path.join(RES, "leakage_check.txt")

I = dict(total=0, marg=1, cop=2, copA=3, copB=4, oaob=5, ap=6, aao=7, bbo=8,
         lplA=9, lplB=10, mcopA=11, mcopB=12, mcop=13)
S, M_, L_ = "rho0.8", "rho0.5", "rho0.2"
LEADS = [S, M_, L_]
AGG = ["avg_60", "avg_240", "max_60", "max_240", "pow_60", "pow_120", "pow_240", "twpow_120", "brier_120"]
CRPS_LIKE = ["site_crps"] + AGG
ORDER_ALL = CRPS_LIKE + ["vs", "es"]


def v(score, rk, term, k="mean"):
    return tab[rk][score][k][I[term]]


def f(x, nd=1, sign=True):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def fa(x, nd=1):
    """absolute value, no sign"""
    return f"{abs(x):.{nd}f}"


def pval(score, rk, term):
    return tab[rk][score]["p"][I[term]]


def fp(p):
    return "P<0.001" if p < 0.001 else (f"P={p:.3f}" if p < 0.01 else f"P={p:.2f}")


def pt(rk, key):
    return pit[rk][key]


has_placebo = len(tab[S]["vs"]["mean"]) > 13
txt = []

# 1. overview and marginal scores --------------------------------------------------------
txt.append(
    r"""Table~\ref{tab:main} and Figure~\ref{fig:decomp} summarise the decomposition. Throughout, differences are A minus B in percent of B's mean score, so negative values favour A. The marginal scores behave as the theory predicts. The site-averaged CRPS favours A by """
    + f"{fa(v('site_crps', S, 'total'))}, {fa(v('site_crps', M_, 'total'))} and {fa(v('site_crps', L_, 'total'))}\\%"
    + r""" at short, medium and long lead, and its copula terms vanish identically, case by case (Proposition~\ref{prop:invariance}a), which is a useful check on any implementation. The energy score behaves almost like a marginal score: its Shapley copula effect never exceeds """
    + f"{max(abs(v('es', rk, 'cop')) for rk in LEADS):.2f}"
    + r"""\% in absolute value, against """
    + f"{min(v('vs', rk, 'cop') for rk in LEADS):.1f}--{max(v('vs', rk, 'cop') for rk in LEADS):.1f}\\%"
    + r""" for the variogram score. With 1,256 components, the energy score is nearly blind to the difference between the two copulas.""")

# 2. aggregated scores ------------------------------------------------------------------------
tot_s = [-v(s, S, 'total') for s in AGG]
txt.append(
    r"""The aggregated scores, by contrast, point the wrong way about dependence. At short lead, the average-pooled, max-pooled and regional-power CRPS, the low-power twCRPS and the wind-drought Brier score all favour A, by """
    + f"{min(tot_s):.1f}--{max(tot_s):.1f}\\%"
    + r""", although A has the worse copula. The decomposition shows why. Marginal effects between """
    + f"{f(max(v(s, S, 'marg') for s in AGG))} and {f(min(v(s, S, 'marg') for s in AGG))}\\%"
    + r""" combine with copula effects that favour B and grow with the pool radius; for regional power the copula effect is """
    + f"{f(v('pow_60', S, 'cop'))}, {f(v('pow_120', S, 'cop'))} and {f(v('pow_240', S, 'cop'))}\\%"
    + r""" at 60, 120 and 240\,km. A's relative advantage in average-pooled CRPS even \emph{increases} with pool size, from """
    + f"{fa(v('avg_60', S, 'total'))}\\% at 60\\,km to {fa(v('avg_240', S, 'total'))}\\% at 240\\,km"
    + r""". This is the pattern that \citet{Price2025} interpret as evidence of better spatial dependence; here it arises because B's offsets are spatially coherent and do not average out.""")

# 3. lead time ---------------------------------------------------------------------------------
txt.append(
    r"""With lead time the copula effects grow in relative terms while the marginal effects shrink. The copula effect on regional power at 240\,km rises from """
    + f"{f(v('pow_240', S, 'cop'))}\\% at short lead to {f(v('pow_240', M_, 'cop'))}\\% and {f(v('pow_240', L_, 'cop'))}\\%"
    + r""" at medium and long lead. For the max-pooled CRPS at 240\,km at long lead, the copula effect ("""
    + f"{f(v('max_240', L_, 'cop'))}\\%) offsets {100 * v('max_240', L_, 'cop') / abs(v('max_240', L_, 'marg')):.0f}\\%"
    + r""" of the marginal effect ("""
    + f"{f(v('max_240', L_, 'marg'))}\\%), and the total ({f(v('max_240', L_, 'total'), 2)}\\%, ${fp(pval('max_240', L_, 'total'))}$)"
    + r""" is indistinguishable from zero. A reading based on totals would declare the two systems equivalent, whereas the decomposition reveals two opposing errors. For the low-power twCRPS the copula effect offsets """
    + f"{100 * v('twpow_120', L_, 'cop') / abs(v('twpow_120', L_, 'marg')):.0f}\\%"
    + r""" of the marginal effect at long lead.""")

# 4. copula-only comparisons, variogram score, independence ------------------------------------
oaob_all = [v(s, rk, 'oaob') for s in AGG + ["vs", "es"] for rk in LEADS]
txt.append(
    r"""Comparisons that hold the marginals fixed recover the correct ranking. On common oracle marginals (column OA$-$OB of Table~\ref{tab:main}), B's copula is better """
    + ("for every aggregated and multivariate score at every lead time" if min(oaob_all) > 0 else "for almost every aggregated and multivariate score")
    + r""", by up to """
    + f"{f(v('max_240', L_, 'oaob'))}\\% for the max-pooled CRPS and {f(v('vs', L_, 'oaob'))}\\% for the variogram score at long lead"
    + r""". Among the comparisons of the raw systems, only the variogram score ranks B significantly ahead of A, at medium and long lead ("""
    + f"{f(v('vs', M_, 'total'))}\\% and {f(v('vs', L_, 'total'))}\\%"
    + r"""), in line with its known sensitivity to dependence \citep{ScheuererHamill2015} and with the weak discrimination of the energy score \citep{PinsonTastu2013}. Destroying A's dependence altogether, by giving it the independence copula (AP), worsens the pooled and regional-power scores by """
    + f"{min(v(s, rk, 'ap') for rk in LEADS for s in ['avg_60', 'pow_60']):.0f}--{max(v(s, rk, 'ap') for rk in LEADS for s in ['max_240', 'pow_240', 'avg_240']):.0f}\\%"
    + r""". Dependence therefore matters a great deal for aggregated quantities, yet at short lead the difference between two \emph{reasonable} copulas is several times smaller than the marginal differences, which is why the aggregated scores are dominated by marginal skill.""")

# 5. leakage in the main experiment ------------------------------------------------------------
if has_placebo:
    SMOOTH = ["avg_60", "avg_240", "pow_60", "pow_120", "pow_240", "twpow_120", "brier_120"]
    smooth_leak = max(abs(v(s, rk, t)) for s in SMOOTH for rk in LEADS for t in ["lplA", "lplB"])
    max_lA = max(v(s, rk, 'lplA') for s in ["max_60", "max_240"] for rk in LEADS)
    vs_lA = [v('vs', rk, 'lplA') for rk in LEADS]
    lB_big = max(abs(v(s, rk, 'lplB')) for s in ["max_60", "max_240", "vs"] for rk in LEADS)
    trip = lambda sc, t: ", ".join(f"{v(sc, rk, t):.2f}" for rk in LEADS[:2]) + f" and {v(sc, L_, t):.2f}"
    txt.append(
        r"""The one-sided effects, and to a lesser extent the Shapley terms, call for the care discussed in Section~\ref{sec:leakage}. Placebo transplants (Table~\ref{tab:placebo} in Appendix~\ref{app:checks}) put the leakage at no more than """
        + f"{smooth_leak:.2f}\\%"
        + r""" in absolute value for the average-pooled and regional-power CRPS, the twCRPS and the Brier score. It is larger for the max-pooled CRPS and the variogram score with A as recipient, up to """
        + f"{max_lA:.1f}\\% and {max(vs_lA):.1f}\\%"
        + r""" respectively, whereas with B as recipient it does not exceed """
        + f"{lB_big:.2f}\\%"
        + r""" in absolute value. Because the two leakages differ, the standard Shapley copula effect understates the copula effect for these two scores (Proposition~\ref{prop:bias}). Matched with placebos, the copula effect of the max-pooled CRPS at 240\,km becomes """
        + trip('max_240', 'mcop') + r"""\% at the three lead times, against """ + trip('max_240', 'cop')
        + r"""\% in Table~\ref{tab:main}, and that of the variogram score becomes """
        + trip('vs', 'mcop') + r"""\%, against """ + trip('vs', 'cop')
        + r"""\%. At long lead, the matched copula effect offsets """
        + f"{100 * v('max_240', L_, 'mcop') / abs(v('max_240', L_, 'total') - v('max_240', L_, 'mcop')):.0f}\\%"
        + r""" of the corresponding marginal effect of the max-pooled CRPS. The correction thus strengthens the conclusions rather than altering them. The matched one-sided effects also clarify a feature of the energy and variogram scores: in B's marginal context their copula effect is negative at short lead, and it remains so after matching ("""
        + f"{f(v('es', S, 'mcopB'), 2)}\\% [{f(v('es', S, 'mcopB', 'lo'), 2)}, {f(v('es', S, 'mcopB', 'hi'), 2)}] and {f(v('vs', S, 'mcopB'), 2)}\\% [{f(v('vs', S, 'mcopB', 'lo'), 2)}, {f(v('vs', S, 'mcopB', 'hi'), 2)}]"
        + r"""). It is therefore a genuine effect of the marginal context, as anticipated by Proposition~\ref{prop:proper}, and not an artefact of the transplant. A plausible explanation is that on B's under-dispersed marginals the weaker dependence of A's copula inflates the differences between sites and so partly compensates for the missing spread.""")

# 6. copula calibration ------------------------------------------------------------------------
txt.append(
    r"""Figure~\ref{fig:pit} shows bivariate Kendall PIT histograms at short lead for 400 pairs of sites 5--100\,km apart and 400 pairs 200--500\,km apart. Because PIT values from overlapping pairs and persistent signals are dependent, Kolmogorov--Smirnov (KS) distances serve as descriptive summaries and are best read relative to those of the oracle ("""
    + f"{pt(S, 'O/near')['ks']:.3f} and {pt(S, 'O/mid')['ks']:.3f}"
    + r"""). B's copula on oracle marginals (OB) is as flat as the oracle ("""
    + f"{pt(S, 'OB/near')['ks']:.3f} and {pt(S, 'OB/mid')['ks']:.3f}"
    + r"""). A's copula on oracle marginals (OA) is $\cup$-shaped, the signature of too little concordance, most clearly for the mid-distance pairs (KS """
    + f"{pt(S, 'OA/mid')['ks']:.3f}, with {100 * pt(S, 'OA/mid')['outer']:.1f}\\%"
    + r""" of the mass in the outer deciles instead of 20\%). That is where A's correlation, $0.95\,e^{-h/150}$ instead of $0.95\,e^{-h/400}$, is most deficient in absolute terms, the largest deficit occurring at about 235\,km. Applied to raw B, the same diagnostic is dominated by B's marginal miscalibration (KS """
    + f"{pt(S, 'BB/near')['ks']:.3f}, with a marginal PIT KS distance of {pt(S, 'mpit/BB')['ks']:.3f} against {pt(S, 'mpit/AA')['ks']:.3f} for A"
    + r""") and would wrongly suggest a defective copula. Copula calibration must therefore be assessed on recalibrated marginals.""")

# 7. sensitivity sweep -----------------------------------------------------------------------
if sweep:
    P = sweep["pow_240"]
    sig_neg = lambda r, k: r[k + "_hi"] < 0
    sig_pos = lambda r, k: r[k + "_lo"] > 0
    mis = [r for r in P if r["ellA"] < 400]
    flips = [r for r in mis if sig_neg(r, "total") and sig_pos(r, "cop")]
    masks = [r for r in mis if not sig_neg(r, "total") and not sig_pos(r, "total") and sig_pos(r, "cop")]
    right = [r for r in mis if sig_pos(r, "total") and sig_pos(r, "cop")]
    tail = {sc: [r for r in sweep[sc] if r["ellA"] == 400] for sc in sweep}
    maxcop = max(abs(r["cop"]) for sc in tail for r in tail[sc])
    t0 = lambda sc: [r for r in tail[sc] if r["tau"] == 0.0][0]
    vsneg = [r["cop"] for r in tail["vs"]]
    vs_short = [r for r in sweep["vs"] if r["ellA"] <= 150]
    vs_short_ok = all(sig_pos(r, "total") for r in vs_short)
    num = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
           8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
    txt.append(
        r"""Figure~\ref{fig:sens} varies B's RMS offset $\sigma_\beta$ between 0 and 0.3 and the range of A's Gaussian copula between 100 and 400\,km, at medium lead and with 300 cases per configuration. Consider first the """
        + num[len(mis)]
        + r""" configurations in which A's range is too short. In """
        + num[len(flips)]
        + r""" of them, the regional-power CRPS at 240\,km significantly favours A while the copula effect significantly favours B, so an inference about dependence drawn from the aggregated score would have the wrong sign. In """
        + num[len(masks)]
        + r""" more, the aggregated score shows no significant difference although B's copula is significantly better, and only in """
        + num[len(right)]
        + r""", those with small offsets and the shortest ranges, does it correctly favour B. """
        + (r"""The variogram score ranks B ahead in every configuration in which A's range is 100 or 150\,km, offsets included. """ if vs_short_ok else "")
        + r"""In the four configurations with a range of 400\,km, A's copula has the same correlation parameters, and hence the same Kendall's tau, as the truth and differs from it only in the tails, where it has no tail dependence. No Shapley copula effect then exceeds """
        + f"{maxcop:.2f}"
        + r"""\% in absolute value. The copula-only comparison detects the difference for the max-pooled CRPS ("""
        + f"{f(t0('max_240')['oaob'], 2)}\\% [{f(t0('max_240')['oaob_lo'], 2)}, {f(t0('max_240')['oaob_hi'], 2)}]"
        + r""" at $\sigma_\beta=0$) and for regional power ("""
        + f"{f(t0('pow_240')['oaob'], 2)}\\% [{f(t0('pow_240')['oaob_lo'], 2)}, {f(t0('pow_240')['oaob_hi'], 2)}]"
        + r"""), but not for the variogram score, the twCRPS or the drought Brier score, whose intervals contain zero. The variogram score's Shapley copula effect is even slightly negative there ("""
        + f"{f(min(vsneg), 1)} to {f(max(vsneg), 1)}\\%"
        + r"""), which may reflect both the marginal context of Proposition~\ref{prop:proper} and finite-ensemble effects. With 50 members and a few hundred cases, a pure tail-dependence error is therefore hard to detect, a sobering result for claims about extremes that rest on joint structure.""")

out = "\n\n".join(txt) + "\n"
os.makedirs(TAB, exist_ok=True)

# ---------------------------------------------------------------------------------------------
# Appendix B text: bivariate leakage check and placebo summary of the main experiment
app = []
if os.path.exists(leak_path):
    L = {}
    for line in open(leak_path):
        parts = line.split()
        sc, M = parts[0], int(parts[1].split("=")[1])
        vals = {}
        for item in parts[2:]:
            k, rest = item.split("=")
            m = re.match(r"([+-]?[0-9.]+)\(([0-9.]+)\)", rest)
            vals[k] = (float(m.group(1)), float(m.group(2)))
        L[(sc, M)] = vals
    vs = {M: L[("vs", M)] for M in (20, 50, 200)}
    su = {M: L[("sum", M)] for M in (20, 50, 200)}

    def within(a, b, k=2.0):
        (ma, ea), (mb, eb) = a, b
        return abs(ma - mb) <= k * (ea ** 2 + eb ** 2) ** 0.5

    def zgap(a, b):
        (ma, ea), (mb, eb) = a, b
        return abs(ma - mb) / (ea ** 2 + eb ** 2) ** 0.5, abs(ma - mb)

    zs_vs = [zgap(L[("vs", M)][x], L[("vs", M)][y]) for M in (20, 50, 200) for x, y in (("LAB", "LplA"), ("LBA", "LplB"))]
    zs_su = [zgap(L[("sum", M)][x], L[("sum", M)][y]) for M in (20, 50, 200) for x, y in (("LAB", "LplA"), ("LBA", "LplB"))]
    zmax_vs = max(z for z, _ in zs_vs)
    gap_su, zgap_su = max((g, z) for z, g in zs_su)
    placebo_sentence = (
        r"""The placebo estimates agree with the directly measured leakages: for the variogram score, where the leakage matters, the largest discrepancy is """
        + f"{zmax_vs:.1f}"
        + r""" standard errors of the difference, and for the CRPS of the sum it is """
        + f"{gap_su:.2f}"
        + r""" percentage points ("""
        + f"{zgap_su:.1f}"
        + r""" standard errors). This supports their use when the leakage cannot be measured directly.""")
    app.append(
        r"""Table~\ref{tab:leak} reports the bivariate check. Because A and B share the same marginals, the hybrid AB has the law of B and the hybrid BA has the law of A, so the leakages $L_{AB}$ and $L_{BA}$ are simply the differences between the scores of the transplanted and the raw ensembles. For the variogram score they are """
        + f"{vs[20]['LAB'][0]:.1f}\\% and {vs[20]['LBA'][0]:.1f}\\% at $M=20$, {vs[50]['LAB'][0]:.1f}\\% and {vs[50]['LBA'][0]:.1f}\\% at $M=50$, and {vs[200]['LAB'][0]:.1f}\\% and {vs[200]['LBA'][0]:.1f}\\% at $M=200$"
        + r""": large compared with the copula effect being measured (a raw difference of """
        + f"{vs[50]['raw'][0]:.2f}\\%"
        + r""" at $M=50$), similar for the two hybrids, and decreasing with $M$. Accordingly, the one-sided effects are biased in opposite directions, by several percent at small $M$, whereas the Shapley copula effect ("""
        + f"{vs[20]['shap'][0]:.2f}, {vs[50]['shap'][0]:.2f} and {vs[200]['shap'][0]:.2f}\\%"
        + r""") stays close to the raw difference ("""
        + f"{vs[20]['raw'][0]:.2f}, {vs[50]['raw'][0]:.2f} and {vs[200]['raw'][0]:.2f}\\%"
        + r""") at every $M$. For the CRPS of the sum the leakage is an order of magnitude smaller, at most """
        + f"{max(abs(su[M][k][0]) for M in (20, 50, 200) for k in ('LAB', 'LBA')):.2f}\\%"
        + r""". """
        + placebo_sentence
    )
if has_placebo:
    vs_l = [v('vs', rk, t) for rk in LEADS for t in ("lplA", "lplB")]
    es_l = [v('es', rk, t) for rk in LEADS for t in ("lplA", "lplB")]
    big = [(s_, rk) for s_ in ORDER_ALL for rk in LEADS if abs(v(s_, rk, 'mcop') - v(s_, rk, 'cop')) > 0.3]
    app.append(
        r"""Table~\ref{tab:placebo} reports the placebo transplants of the main experiment. The leakage is negligible for the site CRPS, for which it vanishes identically, and small for the average-pooled and regional-power CRPS, the twCRPS and the Brier score. For the energy score it lies between """
        + f"{f(min(es_l), 2)} and {f(max(es_l), 2)}\\%"
        + r""", and for the variogram score between """
        + f"{f(min(vs_l), 2)} and {f(max(vs_l), 2)}\\%"
        + r""". Unlike in the bivariate check, the leakage of the max-pooled CRPS and of the variogram score is markedly larger with A as recipient than with B, so it does not cancel in the Shapley average, and the placebo-matched copula effect exceeds the standard one by up to """
        + f"{max(v(s_, rk, 'mcop') - v(s_, rk, 'cop') for s_ in ORDER_ALL for rk in LEADS):.2f}"
        + r""" percentage points. For the other scores the two versions agree to within """
        + f"{max(abs(v(s_, rk, 'mcop') - v(s_, rk, 'cop')) for s_ in ORDER_ALL if s_ not in ('max_60', 'max_240', 'vs') for rk in LEADS):.2f}"
        + r""" percentage points. """
        + (r"""The two versions have the same sign whenever the standard effect is significant, and where they differ noticeably the matched effect is the larger one, so here the leakage makes the standard decomposition conservative."""
           if all(v(s_, rk, 'mcop') >= v(s_, rk, 'cop') for s_, rk in big)
           and all((v(s_, rk, 'mcop') > 0) == (v(s_, rk, 'cop') > 0)
                   for s_ in ORDER_ALL if s_ != "site_crps" for rk in LEADS
                   if v(s_, rk, 'cop', 'lo') > 0 or v(s_, rk, 'cop', 'hi') < 0)
           else r"""The two versions can differ in sign or ordering, so both should be reported."""))
open(os.path.join(TAB, "placebo_text.tex"), "w").write("\n\n".join(app) + "\n")
open(os.path.join(TAB, "results_text.tex"), "w").write(out)
print(out)
print("=====")
print("\n\n".join(app))
