# Separating marginal and dependence skill in machine-learning weather ensembles

Code, input data and results for the paper *Separating marginal and dependence skill in
machine-learning weather ensembles: a simulation study for regional
wind power*.

## Contents

|path|what|size|where to host|
|-|-|-|-|
|`code/`|library, simulation, checks and generators of all tables, figures and quoted numbers (see `code/README.md`)|< 0.1 MB|git|
|`data/gppd\\\_wind.csv`|wind farms of the Global Power Plant Database (5,344 farms)|0.13 MB|git|
|`data/WTK\\\_Validation\\\_IEC-2\\\_normalized.csv`|IEC Class II normalised power curve (NREL)|< 1 kB|git|
|`results/\\\*.json`, `results/leakage\\\_check.txt`, `results/\\\*.log`|summaries behind every table, figure and number in the paper|< 1 MB|git|
|`results/main.npz`|case-level scores (730 cases x 3 lead levels x 10 ensembles x 18 scores) and Kendall PIT values of the main experiment|76 MB|archive (e.g. Zenodo), or git LFS|
|`results/sweep\\\_tau\\\*\\\_ellA\\\*.npz`|case-level scores of the 16 sensitivity runs|6 MB|archive, or git LFS|
|||||
|`DATA\\\_SOURCES.md`|provenance, versions, licences and checksums of the inputs||git|
|`requirements.txt`, `CITATION.cff`, `SHA256SUMS`|environment, citation metadata, checksums of every file||git|

## Format of the result archives

`results/main.npz` (NumPy `.npz`, one array per key):

* `rho{0.8,0.5,0.2}/{ens}/{score}`: 730 case-level scores, for `ens` in `AA, BB, AB, BA, O, AP, OA, OB, PA, PB` (raw systems, transplanted hybrids, oracle, independence copula, copula-only pair on oracle marginals, placebo transplants) and `score` in `site\\\_crps, vs, es, {avg,max,pow,twpow,brier}\\\_{60,120,240}`;
* `rho\\\*/pit/{AA,BB,O,OA,OB}/{near,mid}`: bivariate Kendall PIT values (400 pairs x 730 cases);
* `rho\\\*/mpit/{AA,BB}`: univariate PIT values at 200 sites;
* `meta/\\\*`: geometry and run parameters (number of sites, pool counts, seeds, options).

`results/sweep\\\_tau{sigma\\\_beta}\\\_ellA{range}.npz` have the same `{ens}/{score}` layout for 300 cases at medium lead (no PIT, no placebo).

## Reproducing the paper

```
pip install -r requirements.txt
python -I code/prepare\\\_data.py          # checks the inputs (add --download to rebuild them)
python code/test\\\_copula\\\_verif.py        # unit tests
bash code/run\\\_all.sh                    # about 1 h on 2 cores; rewrites results/, figures/, tables/
latexmk -pdf copula\\\_ml\\\_ensembles.tex
```

With the package versions in `requirements.txt` the pipeline reproduces the archived results
bit for bit (all seeds are fixed). The bootstrap summaries can be recomputed from the archived
`.npz` files without re-running the simulations: `python code/make\\\_figures.py \\\&\\\& python code/make\\\_tables.py \\\&\\\& python code/make\\\_results\\\_text.py`.

## Licences

* Code written for this project:
* `code/icosahedral\\\_mesh.py`: Google LLC, Apache License 2.0 (header retained, unmodified).
* `data/gppd\\\_wind.csv`: derived from the Global Power Plant Database, World Resources
Institute et al., CC BY 4.0 (attribution in `DATA\\\_SOURCES.md`).
* `data/WTK\\\_Validation\\\_IEC-2\\\_normalized.csv`: NREL turbine-models, BSD 3-Clause
(`data/LICENSE-NREL-turbine-models.txt`).

