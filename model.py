"""PCA + K-Means with k chosen by elbow, silhouette AND player-level bootstrap stability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

import config as C


def fit_pca(X: np.ndarray, log=print) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=C.PCA_VARIANCE, svd_solver="full", random_state=C.SEED).fit(X)
    log(f"  PCA: {pca.n_components_} components keep {pca.explained_variance_ratio_.sum():.1%} of variance "
        f"(from {X.shape[1]} features)")
    return pca, pca.transform(X)


def _km(k, seed, n_init=10):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed)


def bootstrap_stability(Z: np.ndarray, players: np.ndarray, k: int, ref: np.ndarray, n_boot: int, seed: int):
    """Resample PLAYERS (all their seasons together) with replacement, refit, predict on the full set,
    ARI vs the reference labels. Resampling rows would let a player's other seasons leak in."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(players)
    idx_by_player = pd.Series(np.arange(len(players))).groupby(players).apply(np.asarray)
    aris = []
    for b in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([idx_by_player[p] for p in pick])
        m = _km(k, seed + b + 1, n_init=4).fit(Z[rows])
        aris.append(adjusted_rand_score(ref, m.predict(Z)))
    return float(np.mean(aris)), float(np.std(aris))


def select_k(Z: np.ndarray, players: np.ndarray, log=print, k_range=None, n_boot=None) -> tuple[int, pd.DataFrame]:
    k_range = k_range or C.K_RANGE
    n_boot = n_boot or C.N_BOOTSTRAP
    rows = []
    sil_n = min(len(Z), 4000)
    sil_idx = np.random.default_rng(C.SEED).choice(len(Z), sil_n, replace=False)
    for k in k_range:
        m = _km(k, C.SEED).fit(Z)
        lab = m.labels_
        seed_aris = [adjusted_rand_score(lab, _km(k, C.SEED + 100 + s, n_init=4).fit_predict(Z)) for s in range(3)]
        st, sd = bootstrap_stability(Z, players, k, lab, n_boot, C.SEED)
        rows.append(dict(k=k, inertia=m.inertia_, silhouette=silhouette_score(Z[sil_idx], lab[sil_idx]),
                         boot_ari=st, boot_ari_sd=sd, seed_ari=float(np.mean(seed_aris)),
                         min_share=np.bincount(lab).min() / len(lab)))
        r = rows[-1]
        log(f"  k={k:2d} inertia={r['inertia']:9.0f} silhouette={r['silhouette']:.3f} "
            f"bootstrap ARI={st:.3f}±{sd:.3f} multi-seed ARI={r['seed_ari']:.3f} smallest={r['min_share']:.1%}")
    res = pd.DataFrame(rows)
    ok = res[(res.boot_ari >= C.STABLE_ARI) & (res.min_share >= C.MIN_CLUSTER_SHARE)]
    if len(ok):
        k = int(ok.k.max())
        log(f"  chosen k={k}: largest k with bootstrap ARI >= {C.STABLE_ARI} and every cluster >= "
            f"{C.MIN_CLUSTER_SHARE:.0%} of rows")
    else:
        k = int(res.loc[res.boot_ari.idxmax(), "k"])
        log(f"  WARNING: no k reached bootstrap ARI {C.STABLE_ARI}; using the most stable k={k}")
    res["chosen"] = res.k == k
    return k, res


def fit_final(Z: np.ndarray, k: int, log=print):
    km = KMeans(n_clusters=k, n_init=50, random_state=C.SEED).fit(Z)
    gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=C.SEED).fit(Z)
    ari = adjusted_rand_score(km.labels_, gmm.predict(Z))
    log(f"  final KMeans k={k}; GaussianMixture(full) agreement ARI={ari:.3f}")
    return km, gmm, ari


def distances(km: KMeans, Z: np.ndarray) -> pd.DataFrame:
    d = km.transform(Z)
    order = np.argsort(d, axis=1)
    d1 = d[np.arange(len(d)), order[:, 0]]
    d2 = d[np.arange(len(d)), order[:, 1]]
    return pd.DataFrame({"dist": d1, "second_cluster": order[:, 1], "ambiguity": d1 / d2})


# ----------------------------------------------------------------------------- naming
def name_clusters(centroids_z: pd.DataFrame, log=print) -> dict[int, str]:
    """Match each cluster's z-profile to an a-priori template (Hungarian on cosine similarity).

    Clusters with no template above MIN_TEMPLATE_COS get a descriptor from their top features.
    The descriptor is always available in `describe()`.
    """
    names = list(C.ARCHETYPE_TEMPLATES)
    sim = np.full((len(centroids_z), len(names)), -1.0)
    for j, t in enumerate(names):
        tf = {f: w for f, w in C.ARCHETYPE_TEMPLATES[t].items() if f in centroids_z.columns}
        if len(tf) < 3:
            continue
        tv = np.array(list(tf.values()), float)
        for i, (_, row) in enumerate(centroids_z.iterrows()):
            cv = row[list(tf)].values.astype(float)
            sim[i, j] = cv @ tv / (np.linalg.norm(cv) * np.linalg.norm(tv) + 1e-9)
    r, c = linear_sum_assignment(-sim)
    out = {}
    for i, j in zip(r, c):
        cid = centroids_z.index[i]
        if sim[i, j] >= C.MIN_TEMPLATE_COS:
            out[cid] = names[j]
        else:
            out[cid] = "Mixed: " + describe(centroids_z.loc[cid], 2)
        log(f"  cluster {cid}: best template '{names[j]}' cos={sim[i, j]:.2f} -> '{out[cid]}' | "
            f"{describe(centroids_z.loc[cid])}")
    for cid in centroids_z.index:            # k > number of templates
        out.setdefault(cid, "Mixed: " + describe(centroids_z.loc[cid], 2))
    return {int(c): n for c, n in out.items()}


LABELS = {"height": "height", "weight": "weight", "usg": "usage", "ast_pct": "assist%", "tov_pct": "TOV%",
          "ast_to": "AST/TO", "orb_pct": "OREB%", "drb_pct": "DREB%", "stl_pct": "steal%", "blk_pct": "block%",
          "pf_100": "fouls", "fg3a_rate": "3PA rate", "fta_rate": "FT rate", "rim_share": "rim (0-3ft) share",
          "short_share": "3-10ft share", "mid_share": "10-16ft share", "long2_share": "long-2 share",
          "corner3_share": "corner-3 share", "avg_shot_dist": "shot distance", "dunk_share": "dunk share",
          "ast_2p": "assisted 2s", "ast_3p": "assisted 3s", "ts": "TS%", "fg3_pct": "3P%", "ft_pct": "FT%",
          "pts_100": "pts/100", "fga_100": "FGA/100", "shoot_foul_drawn_36": "shooting fouls drawn",
          "charges_36": "charges drawn", "blocked_rate": "shots blocked", "bad_pass_36": "bad passes",
          "lost_ball_36": "lost balls"}


def describe(row: pd.Series, n: int = 4) -> str:
    s = row.sort_values()
    hi = [f"+{LABELS.get(f, f)}" for f in s.index[::-1][:n] if s[f] > 0.3]
    lo = [f"-{LABELS.get(f, f)}" for f in s.index[:max(1, n // 2)] if s[f] < -0.3]
    return ", ".join(hi + lo)
