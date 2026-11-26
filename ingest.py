"""Data ingestion -> one analytical base table (ABT) row per player-season.

Sources (all emit the same canonical columns, see config.FEATURE_GROUPS):
    python ingest.py --source kaggle                 # Basketball-Reference export (no live fetch needed)
    python ingest.py --source nba_api                # stats.nba.com via nba_api (USER runs it; cached, resumable)
    python ingest.py --source nba_api --dry-run      # print the call plan and budget only
    python ingest.py --source synthetic              # labelled fake data for offline tests
Output: data/abt_<source>.csv and data/abt_<source>_quality.txt
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

ID_COLS = ["player_id", "player_name", "season", "season_end", "team", "source"]
_log_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _log_lines.append(msg)


def _per36(num, minutes):
    return np.where(minutes > 0, num / minutes * 36, np.nan)


def _safe_div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)


# =============================================================================== Kaggle (B-Ref)
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/sumitrodatta/nba-aba-baa-stats"
KAGGLE_FILES = {
    "adv": ("Advanced.csv", ["season", "lg", "player", "player_id", "team", "pos", "age", "g", "mp", "per",
                             "ts_percent", "x3p_ar", "f_tr", "orb_percent", "drb_percent", "ast_percent",
                             "stl_percent", "blk_percent", "tov_percent", "usg_percent", "ws_48", "obpm",
                             "dbpm", "bpm", "vorp"]),
    "p100": ("Per 100 Poss.csv", ["season", "lg", "player_id", "team", "pts_per_100_poss", "fga_per_100_poss",
                                  "ast_per_100_poss", "tov_per_100_poss", "pf_per_100_poss", "o_rtg", "d_rtg"]),
    "tot": ("Player Totals.csv", ["season", "lg", "player_id", "team", "fga", "x3p", "x3pa", "x3p_percent",
                                  "x2p", "ft", "fta", "ft_percent"]),
    "shoot": ("Player Shooting.csv", ["season", "lg", "player_id", "team", "avg_dist_fga",
                                      "percent_fga_from_x0_3_range", "percent_fga_from_x3_10_range",
                                      "percent_fga_from_x10_16_range", "percent_fga_from_x16_3p_range",
                                      "percent_assisted_x2p_fg", "percent_assisted_x3p_fg",
                                      "percent_dunks_of_fga", "percent_corner_3s_of_3pa"]),
    "pbp": ("Player Play By Play.csv", ["season", "lg", "player_id", "team", "pg_percent", "sg_percent",
                                        "sf_percent", "pf_percent", "c_percent", "on_court_plus_minus_per_100_poss",
                                        "net_plus_minus_per_100_poss", "bad_pass_turnover", "lost_ball_turnover",
                                        "shooting_foul_drawn", "offensive_foul_drawn", "fga_blocked"]),
    "career": ("Player Career Info.csv", ["player_id", "ht_in_in", "wt"]),
}


def download_kaggle(raw_dir: Path) -> Path:
    """Download + unzip the public Kaggle dataset into its own folder (skipped if present)."""
    import requests
    dest = raw_dir / "kaggle_bref"
    if (dest / "Advanced.csv").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    z = raw_dir / "kaggle_bref.zip"
    log(f"downloading {KAGGLE_URL}")
    r = requests.get(KAGGLE_URL, timeout=120)
    log(f"  HTTP {r.status_code}, {len(r.content):,} bytes")
    r.raise_for_status()
    z.write_bytes(r.content)
    import zipfile
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    (raw_dir / "DOWNLOAD_INFO.txt").write_text(f"kaggle sumitrodatta/nba-aba-baa-stats downloaded {time.strftime('%Y-%m-%d')}\n")
    return dest


def _one_row_per_player_season(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """B-Ref lists traded players once per team plus a combined '2TM'/'3TM' row: keep the combined one."""
    combined = df["team"].astype(str).str.fullmatch(r"\d+TM")
    multi = df.duplicated(["player_id", "season"], keep=False)
    teams = (df[~combined].groupby(["player_id", "season"])["team"]
             .agg(lambda s: "/".join(s.astype(str))).rename("teams"))
    out = df[~multi | combined].copy()
    dup = out.duplicated(["player_id", "season"]).sum()
    assert dup == 0, f"{name}: {dup} duplicate player-seasons after keeping combined rows"
    log(f"  {name}: {len(df):,} rows -> {len(out):,} player-seasons ({int(combined.sum()):,} combined traded rows kept, "
        f"{int((multi & ~combined).sum()):,} per-team rows dropped)")
    return out.join(teams, on=["player_id", "season"]) if name == "adv" else out


def load_kaggle(raw_dir: Path, first_end: int, last_end: int) -> pd.DataFrame:
    d = download_kaggle(raw_dir)
    frames = {}
    for key, (fname, cols) in KAGGLE_FILES.items():
        df = pd.read_csv(d / fname)
        missing = sorted(set(cols) - set(df.columns))
        assert not missing, f"{fname}: missing columns {missing} (schema changed?)"
        df = df[cols]
        if "season" in df:
            df = df[(df["lg"] == "NBA") & df["season"].between(first_end, last_end)].drop(columns="lg")
            df = _one_row_per_player_season(df, key)
        frames[key] = df

    abt = frames["adv"]
    log(f"  spine (Advanced): {len(abt):,} player-seasons, seasons {abt.season.min()}-{abt.season.max()}")
    for key in ["p100", "tot", "shoot", "pbp"]:
        f = frames[key].drop(columns="team")
        n0 = len(abt)
        abt = abt.merge(f, on=["player_id", "season"], how="left", validate="1:1", indicator=True)
        miss = int((abt["_merge"] == "left_only").sum())
        log(f"  join {key}: {len(f):,} rows, spine {n0:,} -> {len(abt):,}, {miss:,} spine rows without a match")
        abt = abt.drop(columns="_merge")
    car = frames["career"].drop_duplicates("player_id")
    abt = abt.merge(car, on="player_id", how="left", validate="m:1")
    log(f"  join career (height/weight): {int(abt['ht_in_in'].isna().sum()):,} rows without height")

    mp = abt["mp"].astype(float)
    out = pd.DataFrame({
        "player_id": abt["player_id"], "player_name": abt["player"], "season_end": abt["season"],
        "season": abt["season"].map(C.end_year_to_season), "team": abt["teams"].fillna(abt["team"]),
        "source": "kaggle",
        # body
        "height": abt["ht_in_in"], "weight": abt["wt"],
        # role
        "usg": abt["usg_percent"], "ast_pct": abt["ast_percent"], "tov_pct": abt["tov_percent"],
        "ast_to": _safe_div(abt["ast_per_100_poss"], abt["tov_per_100_poss"]),
        "orb_pct": abt["orb_percent"], "drb_pct": abt["drb_percent"],
        "stl_pct": abt["stl_percent"], "blk_pct": abt["blk_percent"], "pf_100": abt["pf_per_100_poss"],
        # shot profile
        "fg3a_rate": abt["x3p_ar"], "fta_rate": abt["f_tr"],
        "rim_share": abt["percent_fga_from_x0_3_range"], "short_share": abt["percent_fga_from_x3_10_range"],
        "mid_share": abt["percent_fga_from_x10_16_range"], "long2_share": abt["percent_fga_from_x16_3p_range"],
        "corner3_share": abt["percent_corner_3s_of_3pa"], "w_corner3_share": abt["x3pa"],
        "avg_shot_dist": abt["avg_dist_fga"], "dunk_share": abt["percent_dunks_of_fga"],
        "ast_2p": abt["percent_assisted_x2p_fg"], "w_ast_2p": abt["x2p"],
        "ast_3p": abt["percent_assisted_x3p_fg"], "w_ast_3p": abt["x3p"],
        # efficiency
        "ts": abt["ts_percent"], "fg3_pct": abt["x3p_percent"], "w_fg3_pct": abt["x3pa"],
        "ft_pct": abt["ft_percent"], "w_ft_pct": abt["fta"],
        "pts_100": abt["pts_per_100_poss"], "fga_100": abt["fga_per_100_poss"],
        # misc (play-by-play counts -> per 36)
        "shoot_foul_drawn_36": _per36(abt["shooting_foul_drawn"], mp),
        "charges_36": _per36(abt["offensive_foul_drawn"], mp),
        "bad_pass_36": _per36(abt["bad_pass_turnover"], mp),
        "lost_ball_36": _per36(abt["lost_ball_turnover"], mp),
        "blocked_rate": _safe_div(abt["fga_blocked"], abt["fga"]),
        # profile only
        "per": abt["per"], "ws_48": abt["ws_48"], "bpm": abt["bpm"], "obpm": abt["obpm"], "dbpm": abt["dbpm"],
        "vorp": abt["vorp"], "o_rtg": abt["o_rtg"], "d_rtg": abt["d_rtg"],
        "on_court_pm": abt["on_court_plus_minus_per_100_poss"], "on_off_pm": abt["net_plus_minus_per_100_poss"],
        "pos_pg": abt["pg_percent"], "pos_sg": abt["sg_percent"], "pos_sf": abt["sf_percent"],
        "pos_pf": abt["pf_percent"], "pos_c": abt["c_percent"], "pos_label": abt["pos"],
        "g": abt["g"], "mp": mp, "mpg": mp / abt["g"], "age": abt["age"],
    })
    return out


# =============================================================================== nba_api
PT_TYPES = ["Drives", "CatchShoot", "PullUpShot", "Passing", "Rebounding", "Possessions", "Defense"]
PLAY_TYPES = ["Isolation", "PRBallHandler", "PRRollman", "Postup", "Spotup", "Handoff", "Cut",
              "OffScreen", "Transition", "OffRebound"]
DASH_MEASURES = ["Base", "Advanced", "Usage", "Scoring", "Misc"]


def build_call_plan(seasons, season_type="Regular Season", only=None, skip=None):
    """Every request as (group, season, module, class, kwargs). Groups: dash, bio, pt, shotloc, hustle, synergy."""
    plan = []
    st = {"season_type_all_star": season_type}
    for s in seasons:
        plan.append(("dash_Base_Totals", s, "leaguedashplayerstats", "LeagueDashPlayerStats",
                     dict(season=s, measure_type_detailed_defense="Base", per_mode_detailed="Totals", **st)))
        for m in DASH_MEASURES:
            plan.append((f"dash_{m}_Per100", s, "leaguedashplayerstats", "LeagueDashPlayerStats",
                         dict(season=s, measure_type_detailed_defense=m, per_mode_detailed="Per100Possessions", **st)))
        plan.append(("bio", s, "leaguedashplayerbiostats", "LeagueDashPlayerBioStats",
                     dict(season=s, per_mode_simple="Totals", **st)))
        for p in PT_TYPES:
            plan.append((f"pt_{p}", s, "leaguedashptstats", "LeagueDashPtStats",
                         dict(season=s, player_or_team="Player", pt_measure_type=p, per_mode_simple="Totals", **st)))
        plan.append(("shotloc", s, "leaguedashplayershotlocations", "LeagueDashPlayerShotLocations",
                     dict(season=s, distance_range="By Zone", per_mode_detailed="Totals", **st)))
        plan.append(("hustle", s, "leaguehustlestatsplayer", "LeagueHustleStatsPlayer",
                     dict(season=s, per_mode_time="Totals", **st)))
        for p in PLAY_TYPES:
            plan.append((f"synergy_{p}", s, "synergyplaytypes", "SynergyPlayTypes",
                         dict(season=s, player_or_team_abbreviation="P", type_grouping_nullable="offensive",
                              play_type_nullable=p, per_mode_simple="Totals", **st)))

    def family(g):
        return g.split("_")[0]
    if only:
        plan = [c for c in plan if family(c[0]) in only]
    if skip:
        plan = [c for c in plan if family(c[0]) not in skip]
    return plan


def _cache_file(cache: Path, group: str, season: str, season_type: str) -> Path:
    tag = "" if season_type == "Regular Season" else "_" + season_type.replace(" ", "")
    return cache / group / f"{season}{tag}.json"


def fetch_nba_api(seasons, season_type="Regular Season", cache_dir: Path | str = "data/raw/nba_api",
                  only=None, skip=None, pause=0.8, timeout=60, retries=5, proxy=None, dry_run=False):
    """Cache-first, resumable fetch. Each response's raw dict is saved untouched as JSON.

    Existing files are skipped; a call that fails `retries` times is skipped and listed at the end;
    rerunning the same command fetches only what is missing.
    """
    cache = Path(cache_dir)
    plan = build_call_plan(seasons, season_type, only, skip)
    todo = [c for c in plan if not _cache_file(cache, c[0], c[1], season_type).exists()]
    log(f"nba_api plan: {len(plan)} requests over {len(seasons)} seasons, {len(plan) - len(todo)} cached, "
        f"{len(todo)} to fetch (~{len(todo) * 2.5 / 60:.0f}-{len(todo) * 5 / 60:.0f} min at 1.5-4 s/call + {pause}s pause)")
    if dry_run:
        fams = pd.Series([c[0].split("_")[0] for c in todo]).value_counts()
        log("  per group: " + ", ".join(f"{k}={v}" for k, v in fams.items()))
        return [], todo
    failed = []
    for i, (group, season, mod, cls, kw) in enumerate(todo, 1):
        f = _cache_file(cache, group, season, season_type)
        f.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries):
            try:
                Endpoint = getattr(importlib.import_module(f"nba_api.stats.endpoints.{mod}"), cls)
                d = Endpoint(**kw, timeout=timeout, proxy=proxy).get_dict()
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(d))
                tmp.replace(f)                       # atomic: never leave a half-written cache file
                n = sum(len(t.get("rowSet", [])) for t in _result_sets(d))
                log(f"  [{i}/{len(todo)}] {season} {group}: {n} rows")
                break
            except Exception as exc:                 # timeouts, rate limits, malformed responses
                wait = 5 * 2 ** attempt
                log(f"  [{i}/{len(todo)}] retry {attempt + 1}/{retries} {season} {group}: {type(exc).__name__} "
                    f"(waiting {wait}s)")
                time.sleep(wait)
        else:
            failed.append(f"{season} {group}")
        time.sleep(pause)
    if failed:
        log(f"WARNING: {len(failed)} requests failed (rerun the same command to retry only these): "
            + ", ".join(failed))
    return failed, todo


def _result_sets(d: dict) -> list[dict]:
    rs = d.get("resultSets", d.get("resultSet", []))
    return [rs] if isinstance(rs, dict) else list(rs)


def frame_from_dict(d: dict) -> pd.DataFrame:
    """First result set as a DataFrame. Flattens the two-level shot-locations header to 'Zone|STAT'."""
    rs = _result_sets(d)
    if not rs:
        return pd.DataFrame()
    t = rs[0]
    headers = t["headers"]
    if headers and isinstance(headers[0], dict):    # multi-level header (shot locations)
        top = next(h for h in headers if h.get("name") == "SHOT_CATEGORY")
        base = next(h for h in headers if h.get("name") == "columns")["columnNames"]
        skip, span = top["columnsToSkip"], top["columnSpan"]
        cols = base[:skip] + [f"{zone}|{base[skip + j * span + k]}"
                              for j, zone in enumerate(top["columnNames"]) for k in range(span)]
    else:
        cols = headers
    return pd.DataFrame(t["rowSet"], columns=cols)


def _read(cache: Path, group: str, season: str, season_type: str) -> pd.DataFrame | None:
    f = _cache_file(cache, group, season, season_type)
    if not f.exists():
        return None
    return frame_from_dict(json.loads(f.read_text()))


def _dedupe(df: pd.DataFrame, name: str, season: str, sort_col: str | None = None) -> pd.DataFrame:
    """One row per PLAYER_ID. Logs every duplicate; keeps the row with the most minutes/games."""
    dup = df["PLAYER_ID"].duplicated(keep=False)
    if dup.any():
        log(f"    {season} {name}: {int(dup.sum())} rows share {df.loc[dup, 'PLAYER_ID'].nunique()} PLAYER_IDs "
            f"-> keeping the row with the largest {sort_col or 'first'}")
        if sort_col and sort_col in df:
            df = df.sort_values(sort_col, ascending=False)
        df = df.drop_duplicates("PLAYER_ID")
    return df


# canonical name -> (group, source column). Columns for tracking/hustle are read if present and
# reported as missing otherwise (nba_api's expected_data does not list player-mode tracking columns).
NBA_COLUMN_MAP = {
    "dash_Base_Per100": {"pts_100": "PTS", "fga_100": "FGA", "pf_100": "PF", "_ast100": "AST", "_tov100": "TOV"},
    "dash_Advanced_Per100": {"usg": "USG_PCT", "ast_pct": "AST_PCT", "tov_pct": "TM_TOV_PCT", "orb_pct": "OREB_PCT",
                             "drb_pct": "DREB_PCT", "ts": "TS_PCT", "net_rating": "NET_RATING", "o_rtg": "OFF_RATING",
                             "d_rtg": "DEF_RATING", "pie": "PIE"},
    "dash_Usage_Per100": {"stl_pct": "PCT_STL", "blk_pct": "PCT_BLK"},
    "dash_Scoring_Per100": {"ast_2p": "PCT_AST_2PM", "ast_3p": "PCT_AST_3PM", "pts_paint_share": "PCT_PTS_PAINT"},
    "dash_Misc_Per100": {"pts_fb_100": "PTS_FB", "pts_2nd_100": "PTS_2ND_CHANCE"},
    "bio": {"height": "PLAYER_HEIGHT_INCHES", "weight": "PLAYER_WEIGHT"},
    "pt_Drives": {"drives_36": "DRIVES"},
    "pt_CatchShoot": {"cs_fga_36": "CATCH_SHOOT_FGA"},
    "pt_PullUpShot": {"pullup_fga_36": "PULL_UP_FGA"},
    "pt_Passing": {"passes_36": "PASSES_MADE", "potential_ast_36": "POTENTIAL_AST"},
    "pt_Rebounding": {"reb_chances_36": "REB_CHANCES"},
    "pt_Possessions": {"touches_36": "TOUCHES", "sec_per_touch": "AVG_SEC_PER_TOUCH",
                       "drib_per_touch": "AVG_DRIB_PER_TOUCH", "elbow_touch_36": "ELBOW_TOUCHES",
                       "post_touch_36": "POST_TOUCHES", "paint_touch_36": "PAINT_TOUCHES"},
    "pt_Defense": {"def_rim_fga_36": "DEF_RIM_FGA", "def_rim_fg_pct": "DEF_RIM_FG_PCT"},
    "hustle": {"deflections_36": "DEFLECTIONS", "contested_36": "CONTESTED_SHOTS", "screen_ast_36": "SCREEN_ASSISTS",
               "loose_balls_36": "LOOSE_BALLS_RECOVERED", "box_outs_36": "BOX_OUTS", "charges_36": "CHARGES_DRAWN"},
}
PER36_FROM_TOTALS = {c for g in ["pt_Drives", "pt_CatchShoot", "pt_PullUpShot", "pt_Passing", "pt_Rebounding",
                                 "pt_Possessions", "pt_Defense", "hustle"]
                     for c in NBA_COLUMN_MAP[g] if c.endswith("_36")}


def normalize_nba_api(cache_dir: Path | str, seasons, season_type="Regular Season") -> pd.DataFrame:
    """Raw JSON cache -> ABT. Spine = Base Totals (GP/MIN); every other group is left-joined on PLAYER_ID."""
    cache = Path(cache_dir)
    seasons_out, missing_cols = [], {}
    for s in seasons:
        spine = _read(cache, "dash_Base_Totals", s, season_type)
        if spine is None or spine.empty:
            log(f"  {s}: no Base Totals in cache -> season skipped")
            continue
        spine = _dedupe(spine, "dash_Base_Totals", s, "MIN")
        df = pd.DataFrame({
            "player_id": spine["PLAYER_ID"].astype(int), "player_name": spine["PLAYER_NAME"], "season": s,
            "season_end": C.season_to_end_year(s), "team": spine["TEAM_ABBREVIATION"], "source": "nba_api",
            "g": spine["GP"], "mp": spine["MIN"].astype(float), "age": spine["AGE"],
            "fg3a_rate": _safe_div(spine["FG3A"], spine["FGA"]), "fta_rate": _safe_div(spine["FTA"], spine["FGA"]),
            "fg3_pct": _safe_div(spine["FG3M"], spine["FG3A"]), "w_fg3_pct": spine["FG3A"],
            "ft_pct": _safe_div(spine["FTM"], spine["FTA"]), "w_ft_pct": spine["FTA"],
            "_fg2m": spine["FGM"] - spine["FG3M"], "_fg3m": spine["FG3M"],
        })
        df["mpg"] = df["mp"] / df["g"]
        n_spine = len(df)
        for group, cmap in NBA_COLUMN_MAP.items():
            g = _read(cache, group, s, season_type)
            if g is None or g.empty:
                missing_cols.setdefault(group, set()).add(f"{s}: no data")
                continue
            g = _dedupe(g, group, s, "MIN" if "MIN" in g else ("GP" if "GP" in g else None))
            present = {k: v for k, v in cmap.items() if v in g.columns}
            for k, v in cmap.items():
                if v not in g.columns:
                    missing_cols.setdefault(group, set()).add(v)
            sub = g[["PLAYER_ID"] + list(present.values())].rename(columns={v: k for k, v in present.items()})
            sub = sub.rename(columns={"PLAYER_ID": "player_id"})
            sub["player_id"] = sub["player_id"].astype(int)
            for k in present:
                sub[k] = pd.to_numeric(sub[k], errors="coerce")
            matched = sub["player_id"].isin(df["player_id"]).sum()
            if matched < len(sub):
                log(f"    {s} {group}: {len(sub) - matched} players not in Base Totals (dropped)")
            df = df.merge(sub, on="player_id", how="left", validate="1:1")
        assert len(df) == n_spine, "merge changed the spine row count"
        # derived
        if "_ast100" in df:
            df["ast_to"] = _safe_div(df.pop("_ast100"), df.pop("_tov100"))
        for c in PER36_FROM_TOTALS:
            if c in df:
                df[c] = _per36(df[c], df["mp"])
        if "def_rim_fga_36" in df:
            df["w_def_rim_fg_pct"] = df["def_rim_fga_36"] * df["mp"] / 36
        df["w_ast_2p"], df["w_ast_3p"] = df.pop("_fg2m"), df.pop("_fg3m")
        df = df.merge(_shot_locations(cache, s, season_type), on="player_id", how="left", validate="1:1")
        df = df.merge(_synergy(cache, s, season_type), on="player_id", how="left", validate="1:1")
        seasons_out.append(df)
        log(f"  {s}: {n_spine} players in spine")
    for group, cols in missing_cols.items():
        log(f"  NOTE {group}: missing {sorted(cols)}")
    if not seasons_out:
        raise RuntimeError("No nba_api seasons in cache. Run: python ingest.py --source nba_api")
    return pd.concat(seasons_out, ignore_index=True)


def season_presence(abt: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """season x feature: True if > 50% of rows have a value AND the values vary.

    A column that comes back but is constant (e.g. BOX_OUTS = 0 for everyone in 2016-17, before box-out
    tracking) is a placeholder, not data, and counts as absent for that season.
    """
    def ok(d):
        return (d.notna().mean() > 0.5) & (d.nunique() > 1)
    return abt.groupby("season")[feats].apply(ok)


def restrict_to_full_coverage(abt: pd.DataFrame) -> pd.DataFrame:
    """Reconcile features that exist only in some seasons (Synergy/hustle/box-outs start later than tracking).

    - A season missing >= C.MIN_FEATS_TO_DROP_SEASON features is dropped (whole endpoint groups such as
      hustle in 2013-14), so the expensive groups are kept for the remaining seasons.
    - A feature missing in a season that is otherwise complete is dropped instead (one feature is not worth
      a season of rows), and blanked so features.py excludes it.
    - If that would drop more than half of the seasons, all seasons are kept and features.py drops the
      sparse features.
    """
    feats = [f for g in C.FEATURE_GROUPS.values() for f in g if f in abt]
    present = season_presence(abt, feats)
    n_missing = (~present).sum(axis=1)
    bad = sorted(present.index[n_missing >= C.MIN_FEATS_TO_DROP_SEASON])
    n_seasons = present.shape[0]
    if len(bad) > n_seasons / 2:
        log(f"  coverage: {len(bad)}/{n_seasons} seasons lack whole feature groups -> keeping all seasons "
            "(features.py will drop those features)")
        return abt
    for s in bad:
        log(f"  coverage: drop season {s} (no data for {[f for f in feats if not present.loc[s, f]]})")
    abt = abt[~abt["season"].isin(bad)].reset_index(drop=True)
    kept = present.drop(index=bad)
    lone = [f for f in feats if not kept[f].all()]
    for f in lone:
        log(f"  coverage: drop feature {f} (absent or constant in {sorted(kept.index[~kept[f]])}; "
            "too few features missing there to justify dropping the season)")
    return abt.drop(columns=lone)


def _shot_locations(cache, s, season_type) -> pd.DataFrame:
    g = _read(cache, "shotloc", s, season_type)
    if g is None or g.empty:
        return pd.DataFrame({"player_id": pd.Series(dtype=int)})
    g = _dedupe(g, "shotloc", s, None)

    def fga(zone):
        return pd.to_numeric(g.get(f"{zone}|FGA", 0), errors="coerce").fillna(0)
    ra, paint, mid = fga("Restricted Area"), fga("In The Paint (Non-RA)"), fga("Mid-Range")
    corner = fga("Left Corner 3") + fga("Right Corner 3")
    atb = fga("Above the Break 3")
    total = ra + paint + mid + corner + atb + fga("Backcourt")
    return pd.DataFrame({"player_id": g["PLAYER_ID"].astype(int), "rim_share": _safe_div(ra, total),
                         "short_share": _safe_div(paint, total), "mid_share": _safe_div(mid, total),
                         "corner3_share": _safe_div(corner, corner + atb), "w_corner3_share": corner + atb})


def _synergy(cache, s, season_type) -> pd.DataFrame:
    """Play-type frequency = player's possessions of that type / his possessions over all types.

    A play type absent for a player who has other Synergy rows that season means frequency 0.
    Traded players can appear once per team: possessions are summed, then frequencies recomputed.
    """
    parts = []
    for p in PLAY_TYPES:
        g = _read(cache, f"synergy_{p}", s, season_type)
        if g is None or g.empty or "PLAYER_ID" not in g:
            continue
        parts.append(pd.DataFrame({"player_id": g["PLAYER_ID"].astype(int), "pt": f"pt_{p.lower()}",
                                   "poss": pd.to_numeric(g["POSS"], errors="coerce").fillna(0)}))
    cols = [f"pt_{p.lower()}" for p in PLAY_TYPES]
    if not parts:
        return pd.DataFrame(columns=["player_id"] + cols).astype({"player_id": int})
    long = pd.concat(parts)
    wide = long.pivot_table(index="player_id", columns="pt", values="poss", aggfunc="sum", fill_value=0)
    wide = wide.reindex(columns=cols, fill_value=0)
    wide = wide.div(wide.sum(axis=1), axis=0)
    return wide.reset_index()


# =============================================================================== crosswalk (nba_api <-> B-Ref)
NAME_ALIASES = {  # normalised name -> canonical normalised name, applied to BOTH sources
    "ronald holland": "ron holland", "nicolas claxton": "nic claxton", "moe wagner": "moritz wagner",
    "kj martin": "kenyon martin", "xavier tillman sr": "xavier tillman", "cam reddish": "cameron reddish",
    # verified against the 2026-11-01 Kaggle export after the first live nba_api run
    "vincent hunter": "vince hunter", "cam reynolds": "cameron reynolds", "mitchell creek": "mitch creek",
    "vincent edwards": "vince edwards", "ruben nembhard": "rj nembhard", "trevon scott": "tre scott",
    "matt hurt": "matthew hurt", "cui cui": "cui yongxi", "adama bal": "adama alpha bal",
}
# Season-specific B-Ref renames (e.g. "Nigel Hayes" 2018 vs "Nigel Hayes-Davis" 2026, "Jeenathan" vs "Nate"
# Williams) are handled by the PLAYER_ID -> bref_id propagation pass in attach_bref_profile.


def norm_name(s: str) -> str:
    s = str(s).replace("ı", "i").replace("İ", "I").replace("ø", "o").replace("Ø", "O").replace("ł", "l")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`’,]", "", s).replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NAME_ALIASES.get(s, s)


def attach_bref_profile(abt: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Add WS/48, BPM, VORP, PER, position mix from the Kaggle export via (season_end, normalised name)."""
    if not (raw_dir / "kaggle_bref" / "Advanced.csv").exists():
        log("  crosswalk: Kaggle export not present -> no WS/48/BPM/VORP profile columns")
        return abt
    kg = load_kaggle(raw_dir, int(abt.season_end.min()), int(abt.season_end.max()))
    kg["key"] = kg["player_name"].map(norm_name)
    amb = kg.duplicated(["season_end", "key"], keep=False)
    if amb.any():
        log(f"  crosswalk: {int(amb.sum())} ambiguous B-Ref names dropped: {kg.loc[amb, 'player_name'].unique()[:10]}")
    kg = kg[~amb]
    prof = ["per", "ws_48", "bpm", "obpm", "dbpm", "vorp", "on_court_pm", "on_off_pm",
            "pos_pg", "pos_sg", "pos_sf", "pos_pf", "pos_c", "pos_label"]
    right = kg[["season_end", "key", "g"] + prof].rename(columns={"g": "g_bref", "player_id": "bref_id"})
    right["bref_id"] = kg["player_id"]
    a = abt.assign(key=abt["player_name"].map(norm_name)).drop(columns=[c for c in prof if c in abt])
    out = a.merge(right, on=["season_end", "key"], how="left", validate="m:1")
    # pass 2: a PLAYER_ID matched by name in some season reuses that bref_id in its other seasons
    # (B-Ref sometimes lists a player under a different name in different seasons)
    ids = out.dropna(subset=["bref_id"]).groupby("player_id")["bref_id"].agg(lambda s: s.unique())
    ids = ids[ids.map(len) == 1].map(lambda v: v[0])
    miss = out["g_bref"].isna() & out["player_id"].isin(ids.index)
    if miss.any():
        by_id = right.drop(columns="key").drop_duplicates(["season_end", "bref_id"])
        fill = (out.loc[miss, ["season_end", "player_id"]].assign(bref_id=lambda d: d.player_id.map(ids))
                .reset_index().merge(by_id, on=["season_end", "bref_id"], how="inner").set_index("index"))
        out.loc[fill.index, list(by_id.columns.drop("season_end"))] = fill[list(by_id.columns.drop("season_end"))]
        log(f"  crosswalk: {len(fill)} more player-seasons matched through the player's bref_id in other seasons")
    unmatched = out["g_bref"].isna()
    log(f"  crosswalk: {int((~unmatched).sum()):,}/{len(out):,} player-seasons matched to B-Ref "
        f"({(~unmatched).mean():.1%}); unmatched sample: {out.loc[unmatched, 'player_name'].head(15).tolist()}")
    bad = (~unmatched) & ((out["g"] - out["g_bref"]).abs() > 3)
    if bad.any():
        log(f"  crosswalk: {int(bad.sum())} matches disagree on games by >3 (profile columns blanked): "
            f"{out.loc[bad, ['player_name', 'season', 'g', 'g_bref']].head(10).values.tolist()}")
        out.loc[bad, prof] = np.nan
    return out.drop(columns=["key"])


# =============================================================================== synthetic
SYN_ARCHETYPES = {  # z-score style means on a subset of canonical features
    "A": {"height": -1.2, "usg": 1.0, "ast_pct": 1.8, "tov_pct": 0.8, "fg3a_rate": 0.3, "rim_share": -0.5, "stl_pct": 0.6},
    "B": {"height": 0.2, "usg": -0.8, "fg3a_rate": 1.6, "corner3_share": 1.2, "ast_3p": 1.0, "rim_share": -1.0, "ast_pct": -0.5},
    "C": {"height": 1.4, "usg": -0.5, "rim_share": 1.8, "dunk_share": 1.8, "orb_pct": 1.5, "blk_pct": 1.3, "fg3a_rate": -1.4},
    "D": {"height": 1.2, "fg3a_rate": 0.8, "drb_pct": 1.2, "blk_pct": 0.7, "rim_share": -0.3, "ast_pct": -0.3},
    "E": {"height": 0.0, "usg": 1.8, "pts_100": 1.8, "ast_2p": -1.5, "fta_rate": 1.0, "mid_share": 0.8, "ast_pct": 0.6},
    "F": {"height": 0.3, "rim_share": 0.9, "fta_rate": 0.9, "fg3a_rate": -0.9, "stl_pct": 0.5, "short_share": 0.7},
}


def generate_synthetic(n_players=240, seasons=None, seed=7) -> pd.DataFrame:
    """Labelled fake ABT (fictional names, 'true_archetype' column). NOT real NBA data."""
    rng = np.random.default_rng(seed)
    seasons = seasons or C.season_list("2018-19", "2024-25")
    feats = [f for g in ["body", "role", "rebounding", "defense", "shot_profile", "self_creation",
                         "efficiency", "volume"] for f in C.FEATURE_GROUPS[g]]
    keys = list(SYN_ARCHETYPES)
    kinds = rng.choice(keys, n_players)
    rows = []
    for i, k in enumerate(kinds):
        start = rng.integers(0, len(seasons) - 2)
        for j, s in enumerate(seasons[start:]):
            kk = k if rng.random() > 0.08 else rng.choice(keys)       # occasional role change
            mu = SYN_ARCHETYPES[kk]
            r = {f: mu.get(f, 0.0) + rng.normal(0, 0.45) for f in feats}
            r["fg3a_rate"] += 0.15 * j                                   # era drift the pipeline must absorb
            r.update(player_id=f"syn{i:03d}", player_name=f"Synthetic Player {i:03d}", season=s,
                     season_end=C.season_to_end_year(s), team=f"T{i % 30:02d}", source="synthetic",
                     true_archetype=kk, g=int(rng.integers(15, 83)), mp=float(rng.uniform(300, 3000)),
                     age=int(rng.integers(19, 37)), bpm=rng.normal(0, 3), ws_48=rng.normal(0.1, 0.05),
                     vorp=rng.normal(0.5, 1.0), per=rng.normal(15, 4))
            r["mpg"] = r["mp"] / r["g"]
            rows.append(r)
    df = pd.DataFrame(rows)
    for f in ["fg3_pct", "ft_pct", "ast_2p", "ast_3p", "corner3_share"]:
        df[f"w_{f}"] = rng.integers(0, 400, len(df)).astype(float)
    return df


# =============================================================================== common
def filter_and_summarise(abt: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    assert not abt.duplicated(["player_id", "season"]).any(), "duplicate player-season keys in ABT"
    n0 = len(abt)
    keep = (abt["g"] >= C.MIN_GAMES) & (abt["mp"] >= C.MIN_MINUTES)
    abt = abt[keep].reset_index(drop=True)
    log(f"filter g>={C.MIN_GAMES} & mp>={C.MIN_MINUTES}: {n0:,} -> {len(abt):,} player-seasons ({n0 - len(abt):,} removed)")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    abt.to_csv(out_csv, index=False)
    feats = [f for g in C.FEATURE_GROUPS.values() for f in g if f in abt]
    na = abt[feats + [c for c in C.PROFILE_ONLY if c in abt]].isna().mean().sort_values(ascending=False)
    q = [f"ABT {out_csv.name}: {len(abt):,} player-seasons, {abt.player_id.nunique():,} players, "
         f"source={abt.source.iloc[0]}",
         "", "rows per season:", abt.groupby("season").size().to_string(), "",
         "missing share per column (features + profile):", na.round(3).to_string(), "",
         "canonical features not provided by this source: "
         + ", ".join(f for g in C.FEATURE_GROUPS.values() for f in g if f not in abt), "", "ingest log:"] + _log_lines
    out_csv.with_name(out_csv.stem + "_quality.txt").write_text("\n".join(q) + "\n")
    log(f"wrote {out_csv} and {out_csv.stem}_quality.txt")
    return abt


def main(argv=None, workdir: Path | str = C.ROOT) -> pd.DataFrame | None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle", "nba_api", "synthetic"], default="kaggle")
    ap.add_argument("--seasons", nargs=2, default=[C.FIRST_SEASON, C.LAST_SEASON], metavar=("FIRST", "LAST"))
    ap.add_argument("--season-type", default="Regular Season", choices=["Regular Season", "Playoffs"])
    ap.add_argument("--only", nargs="*", help="nba_api groups to fetch: dash bio pt shotloc hustle synergy")
    ap.add_argument("--skip", nargs="*", help="nba_api groups to skip")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--pause", type=float, default=0.8)
    ap.add_argument("--proxy", help="optional proxy URL for nba_api")
    ap.add_argument("--dry-run", action="store_true", help="nba_api: print the call plan and exit")
    ap.add_argument("--no-fetch", action="store_true", help="nba_api: build the ABT from the cache only")
    ap.add_argument("--keep-partial-seasons", action="store_true",
                    help="nba_api: keep seasons where a fetched group (e.g. Synergy) is absent; its features are then dropped")
    a = ap.parse_args(argv)
    _log_lines.clear()
    P = C.paths(workdir, a.source)
    seasons = C.season_list(*a.seasons)

    if a.source == "kaggle":
        abt = load_kaggle(P["raw"], C.season_to_end_year(seasons[0]), C.season_to_end_year(seasons[-1]))
    elif a.source == "nba_api":
        cache = P["raw"] / "nba_api"
        if not a.no_fetch:
            failed, _ = fetch_nba_api(seasons, a.season_type, cache, a.only, a.skip, a.pause, a.timeout,
                                      proxy=a.proxy, dry_run=a.dry_run)
            if a.dry_run:
                return None
        abt = normalize_nba_api(cache, seasons, a.season_type)
        if not a.keep_partial_seasons:
            abt = restrict_to_full_coverage(abt)
        abt = attach_bref_profile(abt, P["raw"])
    else:
        abt = generate_synthetic(seasons=seasons if a.seasons != [C.FIRST_SEASON, C.LAST_SEASON] else None)
    return filter_and_summarise(abt, P["data"] / f"abt_{a.source}.csv")


if __name__ == "__main__":
    main()
