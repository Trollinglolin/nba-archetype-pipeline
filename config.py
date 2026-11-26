"""Central configuration: seasons, thresholds, feature groups, archetype templates."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- seasons
# nba_api uses "2024-25"; the Kaggle/Basketball-Reference export uses the ENDING year (2025).
FIRST_SEASON, LAST_SEASON = "2013-14", "2025-26"   # span with player-tracking data


def season_list(first: str = FIRST_SEASON, last: str = LAST_SEASON) -> list[str]:
    a, b = int(first[:4]), int(last[:4])
    return [f"{y}-{str(y + 1)[2:]}" for y in range(a, b + 1)]


def season_to_end_year(season: str) -> int:
    """'2024-25' -> 2025 (handles the '1999-00' century rollover)."""
    return int(season[:4]) + 1


def end_year_to_season(year: int) -> str:
    """2025 -> '2024-25'."""
    return f"{year - 1}-{str(year)[2:]}"


# ----------------------------------------------------------------------------- filters
MIN_GAMES = 20
MIN_MINUTES = 500

# Empirical-Bayes shrinkage strength (pseudo-attempts at the season league mean) for rates.
SHRINK_K = {"fg3_pct": 100, "ft_pct": 50, "ast_2p": 40, "ast_3p": 25, "corner3_share": 40,
            "def_rim_fg_pct": 60}

WINSOR_Q = (0.01, 0.99)
CORR_PRUNE = 0.90          # drop the later (less interpretable) feature of any pair above this |r|
PCA_VARIANCE = 0.90

# ----------------------------------------------------------------------------- k selection
K_RANGE = range(4, 15)
N_BOOTSTRAP = 20            # player-level bootstrap resamples per k
STABLE_ARI = 0.80           # mean bootstrap ARI required to call a k "stable"
MIN_CLUSTER_SHARE = 0.02    # every cluster must hold >= 2% of player-seasons
SEED = 42

# ----------------------------------------------------------------------------- features
# Canonical feature names shared by every source. Order = interpretability priority:
# when two features correlate above CORR_PRUNE, the one listed LATER is dropped.
FEATURE_GROUPS = {
    "body": ["height", "weight"],
    "role": ["usg", "ast_pct", "tov_pct", "ast_to"],
    "rebounding": ["orb_pct", "drb_pct"],
    "defense": ["stl_pct", "blk_pct", "pf_100"],
    "shot_profile": ["fg3a_rate", "fta_rate", "rim_share", "short_share", "mid_share",
                     "long2_share", "corner3_share", "avg_shot_dist", "dunk_share"],
    "self_creation": ["ast_2p", "ast_3p"],
    "efficiency": ["ts", "fg3_pct", "ft_pct"],
    "volume": ["pts_100", "fga_100"],
    "misc": ["shoot_foul_drawn_36", "charges_36", "blocked_rate", "bad_pass_36", "lost_ball_36",
             "pts_paint_share", "pts_fb_100", "pts_2nd_100"],
    # nba_api-only groups (tracking 2013-14+, hustle/synergy later; coverage checked at run time)
    "tracking": ["drives_36", "cs_fga_36", "pullup_fga_36", "touches_36", "sec_per_touch",
                 "drib_per_touch", "passes_36", "potential_ast_36", "reb_chances_36",
                 "def_rim_fga_36", "def_rim_fg_pct", "elbow_touch_36", "post_touch_36", "paint_touch_36"],
    "hustle": ["deflections_36", "contested_36", "screen_ast_36", "loose_balls_36", "box_outs_36"],
    "playtype": ["pt_isolation", "pt_prballhandler", "pt_prrollman", "pt_postup", "pt_spotup",
                 "pt_handoff", "pt_cut", "pt_offscreen", "pt_transition", "pt_offrebound"],
}

# Never clustered: "how good" axes (would create a quality axis) and position labels
# (would make clusters reproduce positions). Used only to profile/validate archetypes.
PROFILE_ONLY = ["per", "ws_48", "bpm", "obpm", "dbpm", "vorp", "o_rtg", "d_rtg", "net_rating",
                "on_court_pm", "on_off_pm", "pie", "pos_pg", "pos_sg", "pos_sf", "pos_pf", "pos_c",
                "mp", "mpg", "g", "age"]

# A feature group whose coverage within the ABT is below this share is dropped entirely
# (rather than imputed); per-feature missing above this share drops that feature.
MIN_COVERAGE = 0.90
# nba_api: a season is dropped only if at least this many features are absent/constant in it (e.g. all
# 6 hustle features before 2016-17); fewer and the features are dropped instead (see ingest.restrict_to_full_coverage).
MIN_FEATS_TO_DROP_SEASON = 3

# Famous players used ONLY as a post-hoc sanity check (never to tune clusters).
# Kaggle slugs and nba_api PLAYER_IDs.
SANITY_PLAYERS = {
    "Nikola Jokić": ("jokicni01", 203999), "Stephen Curry": ("curryst01", 201939),
    "Rudy Gobert": ("goberru01", 203497), "LeBron James": ("jamesle01", 2544),
    "Draymond Green": ("greendr01", 203110), "Klay Thompson": ("thompkl01", 202691),
    "Chris Paul": ("paulch01", 101108), "Clint Capela": ("capelca01", 203991),
    "Brook Lopez": ("lopezbr01", 201572), "DeMar DeRozan": ("derozde01", 201942),
    "Giannis Antetokounmpo": ("antetgi01", 203507), "P.J. Tucker": ("tuckepj01", 200782),
}

# ----------------------------------------------------------------------------- naming
# Archetype templates written a priori from basketball meaning (directional z-scores),
# NOT from players. Each cluster centroid is matched to one template (Hungarian on cosine
# similarity over the template features that exist); weak matches fall back to a
# descriptor built from the centroid's top features.
ARCHETYPE_TEMPLATES = {
    "Primary Playmaker": {"ast_pct": 2, "usg": 1, "tov_pct": 1, "height": -1, "ast_2p": -1, "potential_ast_36": 2},
    # The two below were added after the first real run: the first-run names ("Primary Playmaker",
    # "3-and-D Wing") overstated usage / defense that those centroids do not show. Clusters are unaffected.
    "Pass-First Guard": {"ast_pct": 1.5, "ast_to": 1.5, "height": -1.5, "usg": -0.5, "charges_36": 0.5, "pts_100": -0.5},
    "Low-Usage Role Wing": {"usg": -1.5, "ast_pct": -1, "corner3_share": 1.5, "ast_2p": 1, "ast_3p": 1, "height": 0.3},
    "Scoring Creator": {"usg": 2, "pts_100": 2, "ast_2p": -1.5, "ast_3p": -1, "fta_rate": 0.5, "pullup_fga_36": 2, "pt_isolation": 2},
    "Movement / Spot-up Shooter": {"fg3a_rate": 2, "ast_3p": 1, "rim_share": -1, "usg": -0.5, "fg3_pct": 1, "cs_fga_36": 2, "pt_spotup": 2, "pt_offscreen": 1},
    "3-and-D Wing": {"fg3a_rate": 1.5, "corner3_share": 1.5, "stl_pct": 1, "usg": -1.5, "ast_3p": 1, "deflections_36": 1},
    "Rim-Running Big": {"height": 1.5, "rim_share": 2, "dunk_share": 2, "orb_pct": 1.5, "fg3a_rate": -1.5, "ast_2p": 1, "pt_prrollman": 2, "pt_cut": 1},
    "Rim Protector": {"height": 1.5, "blk_pct": 2, "drb_pct": 1.5, "usg": -1, "fg3a_rate": -1, "def_rim_fga_36": 2},
    "Stretch Big": {"height": 1.5, "fg3a_rate": 1, "drb_pct": 1, "blk_pct": 0.5, "ast_3p": 1, "corner3_share": -0.5},
    "Playmaking Big": {"height": 1.5, "ast_pct": 1.5, "drb_pct": 1.5, "usg": 1, "pt_postup": 1, "elbow_touch_36": 2},
    "Interior Scorer": {"height": 1, "short_share": 1.5, "mid_share": 0.5, "usg": 1, "fta_rate": 1, "pt_postup": 2, "post_touch_36": 2},
    "Slashing Wing": {"rim_share": 1.5, "fta_rate": 1.5, "fg3a_rate": -1, "height": 0, "drives_36": 2, "pt_transition": 1},
    "Mid-Range Scorer": {"mid_share": 2, "long2_share": 1.5, "fg3a_rate": -1, "usg": 1},
    "Defensive Pest Guard": {"stl_pct": 2, "height": -1, "usg": -1, "charges_36": 1, "deflections_36": 2},
    "Energy Big / Rebounder": {"orb_pct": 2, "drb_pct": 1, "height": 1, "usg": -1.5, "ast_2p": 1, "pt_offrebound": 2, "box_outs_36": 1},
    "Low-Usage Role Guard": {"usg": -1.5, "height": -1, "ast_pct": 0.5, "fg3a_rate": 0.5, "pts_100": -1.5},
}
MIN_TEMPLATE_COS = 0.35

# ----------------------------------------------------------------------------- paths
def paths(workdir: Path | str = ROOT, source: str = "kaggle") -> dict[str, Path]:
    w = Path(workdir)
    return {"raw": w / "data" / "raw", "data": w / "data", "models": w / "models" / source,
            "outputs": w / "outputs" / source}
