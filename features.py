"""ABT -> season-relative, pruned, standardised feature matrix.

Order: coverage check -> shrink rates toward the season mean -> season-median impute ->
winsorise within season -> z-score within season -> correlation prune -> StandardScaler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as C


def candidate_features(abt: pd.DataFrame) -> list[str]:
    feats = [f for g in C.FEATURE_GROUPS.values() for f in g if f in abt.columns]
    return list(dict.fromkeys(feats))


def shrink_rates(df: pd.DataFrame, log) -> pd.DataFrame:
    """Empirical-Bayes shrink: (p*w + m*k)/(w + k) with m the attempt-weighted season mean.

    NaN rates from zero attempts (no 3PA, no made 3s, ...) become the season mean.
    """
    df = df.copy()
    for f, k in C.SHRINK_K.items():
        w = f"w_{f}"
        if f not in df or w not in df:
            continue
        p = pd.to_numeric(df[f], errors="coerce")
        wt = pd.to_numeric(df[w], errors="coerce").fillna(0).where(p.notna(), 0)
        pw = (p.fillna(0) * wt)
        m = pw.groupby(df["season"]).transform("sum") / wt.groupby(df["season"]).transform("sum")
        n_na = int(p.isna().sum())
        df[f] = (pw + m * k) / (wt + k)
        log(f"  shrink {f}: k={k} pseudo-attempts, {n_na} zero-attempt NaNs set to the season mean")
    return df


def season_winsor_z(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    lo, hi = C.WINSOR_Q
    g = df.groupby("season")[feats]
    x = df[feats].clip(g.transform("quantile", lo), g.transform("quantile", hi), axis=0)
    gx = x.groupby(df["season"])
    return (x - gx.transform("mean")) / gx.transform("std").replace(0, np.nan)


def correlation_prune(z: pd.DataFrame, feats: list[str], log) -> list[str]:
    """Greedy in priority order: keep a feature unless it correlates > CORR_PRUNE with one already kept."""
    corr = z[feats].corr().abs()
    kept, dropped = [], []
    for f in feats:
        clash = [k for k in kept if corr.loc[f, k] > C.CORR_PRUNE]
        (dropped.append((f, clash[0], corr.loc[f, clash[0]])) if clash else kept.append(f))
    for f, k, r in dropped:
        log(f"  corr prune: drop {f} (|r|={r:.2f} with {k})")
    return kept


def build_features(abt: pd.DataFrame, log=print):
    """Return (abt_with_absolute_features, X (n x p ndarray), feature list, fitted scaler, z DataFrame)."""
    feats = candidate_features(abt)
    leak = set(feats) & set(C.PROFILE_ONLY)
    assert not leak, f"profile-only columns in features: {leak}"
    df = shrink_rates(abt, log)

    # coverage: whole groups / features missing for many rows (e.g. a season without Synergy) are dropped,
    # not imputed. Rare gaps are imputed with the season median and counted.
    cov = 1 - df[feats].isna().mean()
    low = cov[cov < C.MIN_COVERAGE]
    for f, c in low.items():
        miss_seasons = df.loc[df[f].isna(), "season"].value_counts()
        full = [s for s, n in miss_seasons.items() if n == (df["season"] == s).sum()]
        log(f"  coverage: drop {f} ({c:.0%} of rows present"
            + (f"; absent for whole seasons {sorted(full)} - restrict --seasons to keep it" if full else "") + ")")
    feats = [f for f in feats if f not in low.index]
    n_imp = df[feats].isna().sum()
    for f, n in n_imp[n_imp > 0].items():
        log(f"  impute {f}: {n} rows ({n / len(df):.1%}) set to the season median")
    df[feats] = df[feats].fillna(df.groupby("season")[feats].transform("median"))

    z = season_winsor_z(df, feats)
    # safety net: a feature constant within any season (std 0 -> NaN z) cannot be season-scaled
    bad = {f: sorted(df.loc[z[f].isna(), "season"].unique()) for f in feats if z[f].isna().any()}
    for f, seasons in bad.items():
        log(f"  drop {f}: constant within season(s) {seasons}, cannot be z-scored there")
    const = [f for f in feats if f not in bad and z[f].std() == 0]
    if const:
        log(f"  drop constant features: {const}")
    feats = [f for f in feats if f not in bad and f not in const]
    feats = correlation_prune(z, feats, log)
    scaler = StandardScaler().fit(z[feats].values)
    X = scaler.transform(z[feats].values)
    log(f"  final features ({len(feats)}): {', '.join(feats)}")
    return df, X, feats, scaler, z[feats]
