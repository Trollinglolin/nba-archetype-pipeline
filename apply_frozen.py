"""Assign archetypes to seasons outside the fitted range with the FROZEN Kaggle model (no refit).

The Kaggle model was fitted on 2013-14..2025-26. Every input it uses (Advanced, Per 100, Totals, Shooting,
Play-by-Play, Career Info) exists from 1996-97, so earlier seasons can be labelled with the same preprocessing:
shrink rates toward the season mean -> season-median impute -> winsorise + z-score within season ->
the model's own StandardScaler -> PCA -> K-Means.predict. Nothing is re-estimated across seasons, so labels for a
season do not depend on any other season.

Check: 2014+ seasons are re-labelled through this same path and compared with outputs/kaggle/player_archetypes.csv.

    .venv/bin/python -u apply_frozen.py                     # 1997..2026 -> outputs/kaggle/player_archetypes_1997plus.csv
    .venv/bin/python -u apply_frozen.py --first 1997 --last 2026
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

import config as C
import features as F
import ingest


def assign(first_end: int, last_end: int, workdir=C.ROOT, log=print) -> pd.DataFrame:
    P = C.paths(workdir, "kaggle")
    m = joblib.load(P["models"] / "archetype_model.joblib")
    feats = m["features"]
    abt = ingest.load_kaggle(P["raw"], first_end, last_end)
    abt = abt[(abt["g"] >= C.MIN_GAMES) & (abt["mp"] >= C.MIN_MINUTES)].reset_index(drop=True)
    df = F.shrink_rates(abt, log=lambda *_: None)
    missing = [f for f in feats if f not in df]
    assert not missing, f"frozen features missing from the ABT: {missing}"
    n_imp = int(df[feats].isna().sum().sum())
    df[feats] = df[feats].fillna(df.groupby("season")[feats].transform("median"))
    z = F.season_winsor_z(df, feats).fillna(0.0)
    Z = m["pca"].transform(m["scaler"].transform(z[feats].values))
    lab = m["kmeans"].predict(Z)
    d = m["kmeans"].transform(Z)
    order = np.argsort(d, axis=1)
    names = {int(k): v for k, v in m["names"].items()}
    out = pd.DataFrame({
        "player_id": df["player_id"], "player_name": df["player_name"], "season": df["season"],
        "season_end": df["season_end"], "team": df["team"], "archetype_id": lab,
        "archetype": [names[int(i)] for i in lab],
        "dist_to_centroid": d[np.arange(len(d)), lab].round(3),
        "second_archetype": [names[int(i)] for i in order[:, 1]],
    })
    log(f"[frozen] labelled {len(out):,} player-seasons {first_end}-{last_end}; {n_imp} feature gaps imputed")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1997)
    ap.add_argument("--last", type=int, default=2026)
    a = ap.parse_args(argv)
    out = assign(a.first, a.last)
    P = C.paths(C.ROOT, "kaggle")
    ref = pd.read_csv(P["outputs"] / "player_archetypes.csv")[["player_id", "season_end", "archetype"]]
    chk = out.merge(ref, on=["player_id", "season_end"], suffixes=("", "_fitted"))
    agree = float((chk["archetype"] == chk["archetype_fitted"]).mean())
    print(f"[frozen] agreement with the fitted labels on {len(chk):,} overlapping player-seasons: {agree:.1%}")
    print(pd.crosstab(chk["archetype_fitted"], chk["archetype"]))
    dest = P["outputs"] / f"player_archetypes_{a.first}plus.csv"
    out.to_csv(dest, index=False)
    print(f"[frozen] wrote {dest}")


if __name__ == "__main__":
    main()
