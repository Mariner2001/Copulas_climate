"""Recreate the two input files in ../data from their public sources and check them.

  data/gppd_wind.csv
      All wind farms (primary_fuel == "Wind") of the Global Power Plant Database v1.3.0
      (World Resources Institute, CC BY 4.0), columns latitude, longitude, capacity_mw,
      country.  5,344 farms, 263 GW.  Source:
      https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv
  data/WTK_Validation_IEC-2_normalized.csv
      IEC Class II normalised power curve of the NREL WIND Toolkit validation
      (NREL turbine-models, BSD 3-Clause).  Source:
      https://raw.githubusercontent.com/NREL/turbine-models/master/turbine_models/data/Onshore/WTK_Validation_IEC-2_normalized.csv

Usage:  python -I prepare_data.py [--download]
Without --download the script only verifies the SHA-256 checksums of the files in ../data.
"""
import argparse
import hashlib
import io
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
GPPD_URL = ("https://raw.githubusercontent.com/wri/global-power-plant-database/master/"
            "output_database/global_power_plant_database.csv")
PC_URL = ("https://raw.githubusercontent.com/NREL/turbine-models/master/turbine_models/data/"
          "Onshore/WTK_Validation_IEC-2_normalized.csv")
SHA256 = {
    "gppd_wind.csv": "9a74d2a81d745250eaf40b2a28bfafe655cf4b5307c06962d5c84ee64ea2b8a0",
    "WTK_Validation_IEC-2_normalized.csv": "f717959f0e89926eecdf733cff197007d1d09e60bc02ee91f6230bda0f5b89c8",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    os.makedirs(DATA, exist_ok=True)
    with urllib.request.urlopen(GPPD_URL) as r:
        gppd = pd.read_csv(io.BytesIO(r.read()), low_memory=False)
    wind = gppd[gppd.primary_fuel == "Wind"]
    wind[["latitude", "longitude", "capacity_mw", "country"]].to_csv(
        os.path.join(DATA, "gppd_wind.csv"), index=False)
    with urllib.request.urlopen(PC_URL) as r:
        open(os.path.join(DATA, "WTK_Validation_IEC-2_normalized.csv"), "wb").write(r.read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()
    ok = True
    for name, ref in SHA256.items():
        got = sha256(os.path.join(DATA, name))
        print(f"{name}: {'OK' if got == ref else 'MISMATCH'} ({got})")
        ok &= got == ref
    w = pd.read_csv(os.path.join(DATA, "gppd_wind.csv"))
    print(f"wind farms: {len(w)}, total capacity: {w.capacity_mw.sum() / 1e3:.1f} GW")
    raise SystemExit(0 if ok else 1)
