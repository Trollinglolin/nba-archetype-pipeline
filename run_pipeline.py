"""End-to-end: ingest -> features -> PCA/K-Means -> report.

    python -u run_pipeline.py --source kaggle            # real B-Ref data, no fetch needed
    python -u run_pipeline.py --source nba_api --no-fetch  # after running ingest.py --source nba_api
    python -u run_pipeline.py --source synthetic --workdir /tmp/scratch   # labelled fake data, scratch copy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

import config as C
import features as F
import ingest
import model as M
import report as R


def run(source="kaggle", workdir=C.ROOT, ingest_args=None, skip_ingest=False, k=None, n_boot=None, k_range=None):
    t0 = time.time()
    P = C.paths(workdir, source)
    lines: list[str] = []

    def log(m=""):
        print(m, flush=True)
        lines.append(m)

    abt_path = P["data"] / f"abt_{source}.csv"
    if not skip_ingest or not abt_path.exists():
        log(f"== ingest ({source})")
        abt = ingest.main(["--source", source] + (ingest_args or []), workdir=workdir)
        if abt is None:                      # dry run
            return None
    abt = pd.read_csv(abt_path, dtype={"player_id": str} if source != "nba_api" else None)
    log(f"== features: {len(abt):,} player-seasons")
    abt, X, feats, scaler, z = F.build_features(abt, log)
    log("== model")
    pca, Z = M.fit_pca(X, log)
    if k is None:
        k, res = M.select_k(Z, abt.player_id.values, log, k_range=k_range, n_boot=n_boot)
    else:
        _, res = M.select_k(Z, abt.player_id.values, log, k_range=k_range or [k], n_boot=n_boot)
        res["chosen"] = res.k == k
    km, gmm, gmm_ari = M.fit_final(Z, k, log)
    cent = z.groupby(km.labels_).mean()
    names = M.name_clusters(cent, log)

    P["models"].mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": feats, "scaler": scaler, "pca": pca, "kmeans": km, "names": names},
                P["models"] / "archetype_model.joblib")
    (P["models"] / "features.json").write_text(json.dumps({"features": feats, "k": k, "names": names}, indent=1))
    log("== report")
    pa, summ = R.build_report(abt, z, feats, Z, km, gmm_ari, res, names, pca, P["outputs"], source, lines)
    R.print_console(pa, summ)
    print(f"\ndone in {time.time() - t0:.0f}s -> {P['outputs']}/report.md")
    return pa, summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle", "nba_api", "synthetic"], default="kaggle")
    ap.add_argument("--workdir", default=str(C.ROOT), help="root for data/, models/, outputs/ (use a scratch dir for tests)")
    ap.add_argument("--skip-ingest", action="store_true", help="reuse data/abt_<source>.csv")
    ap.add_argument("--no-fetch", action="store_true", help="nba_api: build from the raw cache only (no network)")
    ap.add_argument("--k", type=int, help="force k (still reports its stability)")
    ap.add_argument("--n-boot", type=int, help=f"bootstrap resamples per k (default {C.N_BOOTSTRAP})")
    a, rest = ap.parse_known_args(argv)
    extra = rest + (["--no-fetch"] if a.no_fetch else [])
    run(a.source, Path(a.workdir), extra, a.skip_ingest, a.k, a.n_boot)


if __name__ == "__main__":
    main()
