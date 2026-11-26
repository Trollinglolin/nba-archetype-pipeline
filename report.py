"""Tables, charts and report.md for a fitted archetype model."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import config as C  # noqa: E402
from model import LABELS, describe  # noqa: E402

# Validated reference palette (dataviz skill): fixed categorical order, blue<->red diverging, gray midpoint.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32a2a"])
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold"})
RADAR_FEATS = ["height", "usg", "ast_pct", "tov_pct", "orb_pct", "drb_pct", "blk_pct", "stl_pct", "fg3a_rate",
               "corner3_share", "rim_share", "mid_share", "fta_rate", "ast_2p", "drives_36", "touches_36"]


def colors_for(k: int) -> list[str]:
    if k <= len(CATEGORICAL):
        return CATEGORICAL[:k]
    print(f"  NOTE: k={k} exceeds the 8 validated categorical slots; extra clusters use tab20 (not CVD-validated)")
    return CATEGORICAL + [matplotlib.colors.to_hex(c) for c in plt.cm.tab20.colors[: k - len(CATEGORICAL)]]


def eta_squared(y: pd.Series, labels: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce")
    m = y.notna()
    y, labels = y[m], labels[m]
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_b = (y.groupby(labels).transform("mean") - y.mean()).pow(2).sum()
    return float(ss_b / ss_tot) if ss_tot else float("nan")


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- charts
def plot_k_selection(res: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    k = res.loc[res.chosen, "k"].iloc[0]
    for a, col, title in [(ax[0], "inertia", "Elbow: within-cluster SS"), (ax[1], "silhouette", "Silhouette")]:
        a.plot(res.k, res[col], color=CATEGORICAL[0], lw=2, marker="o", ms=5)
        a.set_title(title, loc="left")
    ax[2].errorbar(res.k, res.boot_ari, yerr=res.boot_ari_sd, color=CATEGORICAL[0], lw=2, marker="o", ms=5,
                   capsize=3, label="player bootstrap ARI")
    ax[2].plot(res.k, res.seed_ari, color=CATEGORICAL[1], lw=2, marker="s", ms=5, label="multi-seed ARI")
    ax[2].axhline(C.STABLE_ARI, color=MUTED, ls="--", lw=1)
    ax[2].text(res.k.max(), C.STABLE_ARI + 0.01, f"stable ≥ {C.STABLE_ARI}", ha="right", va="bottom", color=INK2)
    ax[2].set_title("Stability", loc="left")
    ax[2].legend(frameon=False, loc="lower left")
    for a in ax:
        a.axvline(k, color=INK2, lw=1, alpha=0.5)
        a.set_xlabel("k")
        a.set_xticks(res.k)
    ax[0].text(k, ax[0].get_ylim()[1], f" chosen k={k}", va="top", color=INK2)
    _save(fig, out / "k_selection.png")


def plot_heatmap(cent: pd.DataFrame, names: dict, out: Path):
    lab = [LABELS.get(c, c) for c in cent.columns]
    fig, ax = plt.subplots(figsize=(max(9, 0.42 * len(lab)), 0.55 * len(cent) + 2.2))
    sns.heatmap(cent.values, cmap=DIVERGING, center=0, vmin=-2, vmax=2, annot=True, fmt=".1f",
                annot_kws={"size": 7}, linewidths=1, linecolor=SURFACE, ax=ax,
                xticklabels=lab, yticklabels=[names[i] for i in cent.index],
                cbar_kws={"label": "mean within-season z-score", "shrink": 0.7})
    ax.set_title("Archetype centroids (season-relative z-scores; 0 = league average that season)", loc="left")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(False)
    _save(fig, out / "centroid_heatmap.png")


def plot_radar(cent: pd.DataFrame, names: dict, cols: list[str], out: Path):
    f = [c for c in RADAR_FEATS if c in cent.columns]
    ang = np.linspace(0, 2 * np.pi, len(f), endpoint=False).tolist()
    n = len(cent)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.5 * nrow), subplot_kw={"polar": True})
    axes = np.atleast_1d(axes).ravel()
    for i, (cid, row) in enumerate(cent.iterrows()):
        a = axes[i]
        v = row[f].clip(-2.5, 2.5).tolist()
        a.plot(ang + ang[:1], [0] * (len(f) + 1), color=MUTED, lw=1, ls="--")
        a.plot(ang + ang[:1], v + v[:1], color=cols[i], lw=2)
        a.fill(ang + ang[:1], v + v[:1], color=cols[i], alpha=0.18)
        a.set_ylim(-2.5, 2.5)
        a.set_yticks([-2, 0, 2])
        a.set_yticklabels(["-2", "0", "+2"], fontsize=6, color=MUTED)
        a.set_xticks(ang)
        short = {"rim_share": "rim share", "mid_share": "mid share", "corner3_share": "corner-3"}
        a.set_xticklabels([short.get(x, LABELS.get(x, x)) for x in f], fontsize=6.5)
        a.set_title(names[cid], fontsize=9.5, pad=14)
    for a in axes[n:]:
        a.set_visible(False)
    fig.suptitle("Archetype profiles (z vs. league that season; dashed ring = average)", x=0.02, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "archetype_profiles.png")


def plot_pca(Z, labels, km, pca, feats, names, cols, out: Path):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Z), min(len(Z), 4000), replace=False)
    load = pd.DataFrame(pca.components_[:2].T, index=feats, columns=["pc1", "pc2"])

    def axis_name(c):
        s = load[c].sort_values()
        return (f"PC{c[-1]} ({pca.explained_variance_ratio_[int(c[-1]) - 1]:.0%}): "
                f"− {LABELS.get(s.index[0], s.index[0])} / + {LABELS.get(s.index[-1], s.index[-1])}")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for cid in sorted(names):
        m = labels[idx] == cid
        ax.scatter(Z[idx][m, 0], Z[idx][m, 1], s=9, color=cols[cid], alpha=0.55, lw=0, label=names[cid])
    cc = km.cluster_centers_
    for cid in sorted(names):
        ax.scatter(cc[cid, 0], cc[cid, 1], s=90, color=cols[cid], edgecolor=SURFACE, lw=2, zorder=3)
        ax.annotate(names[cid], (cc[cid, 0], cc[cid, 1]), xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, color=INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.8))
    ax.set_xlabel(axis_name("pc1"))
    ax.set_ylabel(axis_name("pc2"))
    ax.set_title("Player-seasons on the first two principal components", loc="left")
    ax.legend(frameon=False, fontsize=8, markerscale=2, loc="upper left", bbox_to_anchor=(1, 1))
    _save(fig, out / "pca_scatter.png")


def plot_share_by_season(pa: pd.DataFrame, names: dict, cols: list[str], out: Path):
    sh = pd.crosstab(pa.season, pa.archetype_id, normalize="index") * 100
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for cid in sh.columns:
        ax.plot(sh.index, sh[cid], color=cols[cid], lw=2, marker="o", ms=4, label=names[cid])
    ax.set_ylabel("% of qualified player-seasons")
    ax.set_title("How common each archetype is, by season", loc="left")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, out / "archetype_share_by_season.png")


def transitions(pa: pd.DataFrame, names: dict, out: Path) -> tuple[pd.DataFrame, float]:
    p = pa[["player_id", "season_end", "archetype"]].sort_values(["player_id", "season_end"])
    nxt = p.groupby("player_id").shift(-1)
    pair = p.assign(next_arch=nxt["archetype"], next_season=nxt["season_end"])
    pair = pair[pair.next_season == pair.season_end + 1]
    order = [names[i] for i in sorted(names)]
    tm = pd.crosstab(pair.archetype, pair.next_arch, normalize="index").reindex(index=order, columns=order).fillna(0)
    stay = float((pair.archetype == pair.next_arch).mean())
    tm.round(3).to_csv(out / "transitions.csv")
    fig, ax = plt.subplots(figsize=(1.1 * len(order) + 3, 0.7 * len(order) + 1.5))
    blues = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    sns.heatmap(tm * 100, cmap=blues, vmin=0, vmax=100, annot=True, fmt=".0f", linewidths=1, linecolor=SURFACE,
                ax=ax, cbar_kws={"label": "% of next-season rows"})
    ax.set_xlabel("archetype next season")
    ax.set_ylabel("archetype this season")
    ax.set_title(f"Season-to-season transitions ({len(pair):,} consecutive pairs; {stay:.0%} stay put)", loc="left")
    ax.grid(False)
    _save(fig, out / "transitions.png")
    return tm, stay


# ----------------------------------------------------------------------------- tables + report
def build_report(abt, z, feats, Z, km, gmm_ari, res, names, pca, out: Path, source: str, extra_log: list[str]):
    out.mkdir(parents=True, exist_ok=True)
    k = km.n_clusters
    cols = colors_for(k)
    labels = km.labels_
    from model import distances
    dd = distances(km, Z)
    pa = abt[["player_id", "player_name", "season", "season_end", "team"]].copy()
    pa["archetype_id"] = labels
    pa["archetype"] = [names[i] for i in labels]
    pa["dist_to_centroid"] = dd["dist"].round(3).values
    pa["second_archetype"] = [names[i] for i in dd["second_cluster"]]
    pa["ambiguity"] = dd["ambiguity"].round(3).values      # d1/d2: 1.0 = exactly between two archetypes
    if "true_archetype" in abt:
        pa["true_archetype"] = abt["true_archetype"].values
    pa.to_csv(out / "player_archetypes.csv", index=False)

    cent = z.groupby(labels).mean()
    cent.round(3).to_csv(out / "centroids_z.csv")
    prof_cols = [c for c in ["mpg", "mp", "usg", "height", "fg3a_rate", "ast_pct", "per", "ws_48", "bpm", "vorp",
                             "on_off_pm", "pos_pg", "pos_sg", "pos_sf", "pos_pf", "pos_c"] if c in abt]
    summ = abt[prof_cols].groupby(labels).mean()
    summ.insert(0, "share", pd.Series(labels).value_counts(normalize=True).sort_index())
    summ.insert(0, "n", pd.Series(labels).value_counts().sort_index())
    summ.insert(0, "archetype", [names[i] for i in summ.index])
    summ["profile"] = [describe(cent.loc[i]) for i in summ.index]
    summ.round(3).to_csv(out / "cluster_summary.csv", index_label="archetype_id")

    eta = {c: eta_squared(abt[c], pd.Series(labels, index=abt.index))
           for c in ["mp", "mpg", "usg", "bpm", "ws_48", "vorp", "height"] if c in abt}

    plot_k_selection(res, out)
    plot_heatmap(cent, names, out)
    plot_radar(cent, names, cols, out)
    plot_pca(Z, labels, km, pca, feats, names, cols, out)
    plot_share_by_season(pa, names, cols, out)
    tm, stay = transitions(pa, names, out)

    # representatives / ambiguous
    reps, amb = [], []
    for cid in sorted(names):
        sub = pa[pa.archetype_id == cid]
        reps.append(sub.nsmallest(8, "dist_to_centroid").assign(kind="representative"))
        amb.append(sub.nlargest(5, "ambiguity").assign(kind="ambiguous"))
    ra = pd.concat(reps + amb)
    ra.to_csv(out / "representatives_and_ambiguous.csv", index=False)

    # post-hoc sanity check with famous players (never used for fitting)
    sanity = None
    if source != "synthetic":
        idcol = 0 if source == "kaggle" else 1
        ids = {v[idcol]: n for n, v in C.SANITY_PLAYERS.items()}
        s = pa[pa.player_id.isin(ids)]
        if len(s):
            sanity = s.pivot_table(index="player_name", columns="season", values="archetype", aggfunc="first")
            sanity.to_csv(out / "sanity_players.csv")

    write_markdown(out, source, abt, feats, pca, res, k, gmm_ari, summ, eta, stay, pa, ra, sanity, extra_log)
    return pa, summ


def _md_table(df: pd.DataFrame, floatfmt=".2f") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    head = "| " + " | ".join(map(str, [df.index.name or ""] + list(df.columns))) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join(map(str, [i] + list(r))) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join([head, sep] + rows)


def write_markdown(out, source, abt, feats, pca, res, k, gmm_ari, summ, eta, stay, pa, ra, sanity, extra_log):
    L = []
    if source == "synthetic":
        L += ["> **SYNTHETIC DATA - NOT REAL NBA PLAYERS.** Fictional names; generated only to test the pipeline.", ""]
    src = {"kaggle": "Basketball-Reference season tables via the Kaggle export `sumitrodatta/nba-aba-baa-stats` "
                     "(player_id = B-Ref slug)",
           "nba_api": "stats.nba.com via `nba_api` (player_id = NBA PLAYER_ID); WS/48/BPM/VORP joined from the "
                      "Kaggle export by (season, normalised name)",
           "synthetic": "synthetic generator in ingest.py"}[source]
    L += [f"# NBA player archetypes ({source})", "",
          f"- Data: {src}", f"- Player-seasons: {len(abt):,} ({abt.player_id.nunique():,} players), "
          f"seasons {abt.season.min()} to {abt.season.max()}, filter g>={C.MIN_GAMES} and mp>={C.MIN_MINUTES}",
          f"- Clustering inputs ({len(feats)} season-relative z-scores): {', '.join(feats)}",
          f"- PCA: {pca.n_components_} components ({pca.explained_variance_ratio_.sum():.1%} variance)",
          f"- Not used as inputs (profiling only): {', '.join(c for c in C.PROFILE_ONLY if c in abt)}", "",
          "## Choosing k", "",
          f"Chosen **k = {k}**: the largest k whose player-level bootstrap ARI is >= {C.STABLE_ARI} with every "
          f"cluster >= {C.MIN_CLUSTER_SHARE:.0%} of rows. Silhouette is low across the board (player styles form a "
          "continuum, not separated blobs), so it is a weak guide here; the elbow is gradual. "
          "Stability decides.", "",
          _md_table(res.set_index("k")[["inertia", "silhouette", "boot_ari", "boot_ari_sd", "seed_ari", "min_share"]],
                    ".3f"), "",
          f"GaussianMixture (full covariance) vs K-Means agreement at k={k}: ARI = {gmm_ari:.3f}.", "",
          "![k selection](k_selection.png)", "",
          "## Archetypes", "",
          "Names come from matching each centroid's z-profile to templates written in advance from basketball "
          "meaning (config.ARCHETYPE_TEMPLATES); the `profile` column lists the strongest deviations.", "",
          _md_table(summ.set_index("archetype")), "",
          "![centroids](centroid_heatmap.png)", "", "![profiles](archetype_profiles.png)", "",
          "![pca](pca_scatter.png)", "",
          "## Are clusters just playing time or quality?", "",
          "η² = share of a variable's variance explained by archetype membership (0 = unrelated, 1 = determined).", "",
          "| variable | η² |", "|---|---|"] + [f"| {c} | {v:.3f} |" for c, v in eta.items()] + [
          "", "Read mp/mpg η² as 'how much archetype is a minutes proxy'; bpm/ws_48/vorp were never inputs and show "
          "how much the style groups differ in value.", "",
          "## Movement between seasons", "",
          f"{stay:.0%} of consecutive player-season pairs keep the same archetype.", "",
          "![transitions](transitions.png)", "", "![share by season](archetype_share_by_season.png)", "",
          "## Representatives (closest to centroid) and most ambiguous players", ""]
    for a in summ.archetype:
        r = ra[(ra.archetype == a) & (ra.kind == "representative")]
        m = ra[(ra.archetype == a) & (ra.kind == "ambiguous")]
        L.append(f"- **{a}**: " + ", ".join(f"{n} ({s})" for n, s in zip(r.player_name, r.season)))
        L.append("  - borderline: " + ", ".join(f"{n} ({s}, vs {b})"
                                                 for n, s, b in zip(m.player_name, m.season, m.second_archetype)))
    if sanity is not None:
        L += ["", "## Post-hoc sanity check (famous players; never used to fit or name clusters)", "",
              _md_table(sanity.fillna("·"))]
    if "true_archetype" in pa:
        from sklearn.metrics import adjusted_rand_score
        L += ["", f"Recovery of the planted synthetic archetypes: ARI = "
                  f"{adjusted_rand_score(pa.true_archetype, pa.archetype_id):.3f}"]
    L += ["", "## Pipeline log", "", "```"] + extra_log + ["```"]
    (out / "report.md").write_text("\n".join(L) + "\n")


def print_console(pa: pd.DataFrame, summ: pd.DataFrame):
    for _, r in summ.iterrows():
        sub = pa[pa.archetype == r.archetype]
        reps = sub.nsmallest(5, "dist_to_centroid")
        amb = sub.nlargest(3, "ambiguity")
        print(f"\n[{r.archetype}] n={int(r.n)} ({r.share:.0%}) | {r.profile}")
        print("   closest to centroid: " + "; ".join(f"{a} {b}" for a, b in zip(reps.player_name, reps.season)))
        print("   most ambiguous:      " + "; ".join(f"{a} {b} (vs {c})"
                                                     for a, b, c in zip(amb.player_name, amb.season, amb.second_archetype)))
