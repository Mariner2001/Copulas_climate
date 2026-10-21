# Code for "Separating marginal and dependence skill in machine-learning weather ensembles"

Everything in the paper (both tables in the main text and the appendix, all figures, and every
number quoted in Section 7.2 and Appendix B) is produced by the scripts in this folder.

```
python -I prepare_data.py            # verify the two input files (add --download to rebuild them)
python test_copula_verif.py          # unit tests of the library
./run_all.sh                         # full pipeline, about 1 h on 2 cores
```

| file | content |
|---|---|
| `copula_verif.py` | library: `transplant` (copula transplantation / ECC-type reordering), `shuffle_independent`, fair `crps_fair`, `twcrps_fair`, `energy_score_fair`, `variogram_score`, ensemble Kendall PIT (`coppit`, `coppit_pairs`, `marginal_pit`), `shapley_2x2`, moving-block bootstrap intervals and p-values, Benjamini–Yekutieli |
| `test_copula_verif.py` | unit tests, including the exact identity H_T(y) = C_z(F_x(y)) of Proposition 1 and the exact uniformity of the Kendall PIT (Lemma 1) |
| `simulate_wind.py` | simulation at the 1,256 CONUS wind-farm sites of GPPD with the IEC Class II power curve (Section 7); placebo transplants on a separate random stream |
| `leakage_check.py` | bivariate finite-ensemble check of the transplant (Section 4.2, Table in Appendix B) |
| `make_figures.py` | figures and JSON summaries (column order documented in the file header) |
| `make_tables.py` | LaTeX tables, with Benjamini–Yekutieli control over the 105 non-degenerate tests |
| `make_results_text.py` | LaTeX text of Section 7.2 and Appendix B, with every number read from the JSON outputs |
| `prepare_data.py` | rebuilds and checksums the two input files from their public sources |
| `run_all.sh` | runs the whole pipeline |
| `icosahedral_mesh.py` | unmodified mesh code from google-deepmind/graphcast (WeatherNext), Apache-2.0, used for the pool centres |

Random seeds are fixed (main experiment 20261019, placebo stream 20261020, sweep 777,
bivariate check 5 and 6), so the results are reproducible bit for bit with the package
versions in `requirements.txt`.

## Applying the decomposition to GenCast vs ENS
For each case (common valid time, common truth), load the two 50-member ensembles as arrays
of shape `(50, d)` (sites or grid points) and the verifying field `(d,)`, then:

```python
import copula_verif as cv
AB = cv.transplant(gencast, ens)       # GenCast marginals + ENS copula
BA = cv.transplant(ens, gencast)       # ENS marginals + GenCast copula
PA = cv.transplant(gencast, gencast2)  # placebo: second, independent GenCast ensemble
s = {k: score(X, obs) for k, X in dict(AA=gencast, BB=ens, AB=AB, BA=BA, PA=PA).items()}
terms = cv.shapley_2x2(s["AA"], s["AB"], s["BA"], s["BB"])
```

Collect the case-level `terms` over initialisations and call
`cv.block_bootstrap_ci(np.c_[terms["total"], terms["marg"], terms["cop"]], L=10)`.
For ENS, a placebo can be formed from two halves of the ensemble (25 + 25 members).
