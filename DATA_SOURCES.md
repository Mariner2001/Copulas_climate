# Input data: provenance and licences

| file | source | version | licence | SHA-256 |
|---|---|---|---|---|
| `data/gppd_wind.csv` | Global Power Plant Database, World Resources Institute: all plants with `primary_fuel == "Wind"`, columns `latitude, longitude, capacity_mw, country` (5,344 farms, 263 GW) | GitHub copy of the database (`output_database/`, 34,936 plants); same 5,344 wind farms as the v1.3.0 release (June 2021) used by Price et al. (2025) | CC BY 4.0 (attribution required) | `9a74d2a81d745250eaf40b2a28bfafe655cf4b5307c06962d5c84ee64ea2b8a0` |
| `data/WTK_Validation_IEC-2_normalized.csv` | NREL `turbine-models`, `turbine_models/data/Onshore/WTK_Validation_IEC-2_normalized.csv` (IEC Class II normalised power curve of the WIND Toolkit validation; King et al. 2014) | repository `master`, retrieved 2026-10 | BSD 3-Clause (licence text in `data/LICENSE-NREL-turbine-models.txt`) | `f717959f0e89926eecdf733cff197007d1d09e60bc02ee91f6230bda0f5b89c8` |

`code/prepare_data.py --download` rebuilds both files from these public URLs and checks the checksums:

- https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv
- https://raw.githubusercontent.com/NREL/turbine-models/master/turbine_models/data/Onshore/WTK_Validation_IEC-2_normalized.csv

The path cited by Price et al. (2025), `NREL/turbine-models/blob/master/Onshore/WTK_Validation_IEC-2_normalized.csv`, no longer exists; the file now lives under `turbine_models/data/Onshore/` with identical content.

Attribution for GPPD: Global Energy Observatory, Google, KTH Royal Institute of Technology in Stockholm, Enipedia, World Resources Institute. 2018. Global Power Plant Database. Published on Resource Watch and Google Earth Engine; http://resourcewatch.org/ https://earthengine.google.com/

`code/icosahedral_mesh.py` is unmodified code from google-deepmind/graphcast (WeatherNext), Apache License 2.0; its licence header is retained.

The official v1.3.0 release is distributed at https://datasets.wri.org/dataset/globalpowerplantdatabase ; the GitHub README describes the copy in `output_database/` as the latest version of the database. The wind-farm subset used here has the same size (5,344 farms) as the one reported by Price et al. (2025).
