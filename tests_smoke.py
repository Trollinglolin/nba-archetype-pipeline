"""Offline smoke tests (no network, ~30 s):  .venv/bin/python -W ignore tests_smoke.py"""
from __future__ import annotations

import importlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import ingest

ROOT = Path(__file__).resolve().parent


def test_season_mapping():
    assert C.season_to_end_year("2024-25") == 2025 and C.end_year_to_season(2025) == "2024-25"
    assert C.season_to_end_year("1999-00") == 2000 and C.end_year_to_season(2000) == "1999-00"
    assert C.season_list("2013-14", "2025-26")[0] == "2013-14" and len(C.season_list()) == 13


def test_norm_name():
    assert ingest.norm_name("Nikola Jokić") == ingest.norm_name("Nikola Jokic") == "nikola jokic"
    assert ingest.norm_name("Jaren Jackson Jr.") == "jaren jackson"
    assert ingest.norm_name("Shai Gilgeous-Alexander") == "shai gilgeous alexander"
    assert ingest.norm_name("Ronald Holland II") == "ron holland"


# ----------------------------------------------------------------------------- fake nba_api
def _rs(name, headers, rows):
    return {"resource": name, "resultSets": [{"name": name, "headers": headers, "rowSet": rows}]}


def fake_response(cls, kw):
    """Correctly-shaped raw dicts for 3 players (player 3 is traded: two Synergy team rows; bio duplicated)."""
    P = [(1, "Guard One", 25), (2, "Big Two", 28), (3, "Wing Three", 23)]
    if cls == "LeagueDashPlayerStats":
        m = kw["measure_type_detailed_defense"]
        cols = {"Base": ["GP", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PTS", "AST", "TOV", "PF"],
                "Advanced": ["GP", "MIN", "USG_PCT", "AST_PCT", "TM_TOV_PCT", "OREB_PCT", "DREB_PCT", "TS_PCT",
                             "NET_RATING", "OFF_RATING", "DEF_RATING", "PIE"],
                "Usage": ["GP", "MIN", "PCT_STL", "PCT_BLK"],
                "Scoring": ["GP", "MIN", "PCT_AST_2PM", "PCT_AST_3PM", "PCT_PTS_PAINT"],
                "Misc": ["GP", "MIN", "PTS_FB", "PTS_2ND_CHANCE"]}[m]
        h = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE"] + cols
        rows = [[i, n, "AAA", a] + [60 + i, 1500.0 + 100 * i] + [float(10 * i + j) for j in range(len(cols) - 2)]
                for i, n, a in P]
        return _rs("LeagueDashPlayerStats", h, rows)
    if cls == "LeagueDashPlayerBioStats":
        h = ["PLAYER_ID", "PLAYER_NAME", "GP", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT"]
        rows = [[i, n, 60, 74 + 4 * i, str(190 + 20 * i)] for i, n, _ in P] + [[3, "Wing Three", 10, 80, "230"]]
        return _rs("LeagueDashPlayerBioStats", h, rows)
    if cls == "LeagueDashPtStats":
        h = ["PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "DRIVES", "CATCH_SHOOT_FGA", "TOUCHES"]
        return _rs("LeagueDashPtStats", h, [[i, n, 60, 1500.0, 100.0 * i, 50.0, 1000.0] for i, n, _ in P])
    if cls == "LeagueDashPlayerShotLocations":       # two-level header, as stats.nba.com returns it
        zones = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range", "Left Corner 3", "Right Corner 3",
                 "Above the Break 3", "Backcourt"]
        base = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE"] + ["FGM", "FGA", "FG_PCT"] * 7
        rows = [[i, n, 1, "AAA", a] + [v for z in range(7) for v in (5, [40, 20, 20, 5, 5, 10, 0][z] * i, 0.5)]
                for i, n, a in P]
        return {"resource": "leaguedashplayershotlocations", "resultSets": {
            "name": "ShotLocations", "rowSet": rows,
            "headers": [{"name": "SHOT_CATEGORY", "columnSpan": 3, "columnsToSkip": 5, "columnNames": zones},
                        {"name": "columns", "columnSpan": 1, "columnNames": base}]}}
    if cls == "LeagueHustleStatsPlayer":
        h = ["PLAYER_ID", "PLAYER_NAME", "G", "MIN", "DEFLECTIONS", "CONTESTED_SHOTS", "SCREEN_ASSISTS",
             "LOOSE_BALLS_RECOVERED", "BOX_OUTS", "CHARGES_DRAWN"]
        return _rs("HustleStatsPlayer", h, [[i, n, 60, 1500.0, 90, 200, 30, 20, 40, 3] for i, n, _ in P])
    if cls == "SynergyPlayTypes":
        h = ["SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "PLAY_TYPE", "POSS"]
        pt = kw["play_type_nullable"]
        rows = [["22023", 1, "Guard One", "AAA", pt, 100.0], ["22023", 3, "Wing Three", "AAA", pt, 30.0],
                ["22023", 3, "Wing Three", "BBB", pt, 20.0]]
        if pt == "Postup":
            rows.append(["22023", 2, "Big Two", "AAA", pt, 200.0])   # Big Two only has post-ups
        return _rs("SynergyPlayType", h, rows)
    raise AssertionError(cls)


def test_nba_api_path():
    calls = {"n": 0}
    patched = []
    for _, _, mod, cls, _ in ingest.build_call_plan(["2023-24"]):
        m = importlib.import_module(f"nba_api.stats.endpoints.{mod}")
        if (m, cls) in [(p[0], p[1]) for p in patched]:
            continue

        def make(cls_name):
            class Fake:
                def __init__(self, timeout=None, proxy=None, **kw):
                    calls["n"] += 1
                    if calls["n"] == 3:
                        raise TimeoutError("simulated timeout")
                    self.kw = kw

                def get_dict(self):
                    return fake_response(cls_name, self.kw)
            return Fake
        patched.append((m, cls, getattr(m, cls)))
        setattr(m, cls, make(cls))
    orig_sleep, ingest.time.sleep = ingest.time.sleep, lambda *_: None
    try:
        d = Path(tempfile.mkdtemp())
        plan = ingest.build_call_plan(["2023-24"])
        assert len(plan) == 26, len(plan)
        failed, todo = ingest.fetch_nba_api(["2023-24"], cache_dir=d, pause=0)
        assert not failed and len(todo) == 26 and calls["n"] == 27           # one retried timeout
        assert len(list(d.rglob("*.json"))) == 26
        n = calls["n"]
        failed, todo = ingest.fetch_nba_api(["2023-24"], cache_dir=d, pause=0)
        assert calls["n"] == n and not todo                                 # rerun: zero calls

        abt = ingest.normalize_nba_api(d, ["2023-24"])
        assert not abt.duplicated(["player_id", "season"]).any() and len(abt) == 3   # merge key unique
        r1 = abt.set_index("player_id")
        assert abs(r1.loc[1, "rim_share"] - 40 / 100) < 1e-9                 # flattened shot-location header
        assert abs(r1.loc[1, "corner3_share"] - 10 / 20) < 1e-9
        assert r1.loc[2, "pt_postup"] == 1.0 and r1.loc[2, "pt_spotup"] == 0.0   # absent play type = 0
        assert abs(r1.loc[3, "pt_isolation"] - 0.1) < 1e-9                    # traded: 50 of 500 possessions
        assert r1.loc[3, "height"] in (86.0, 80.0) and r1.loc[2, "weight"] == 230.0
        assert abs(r1.loc[1, "drives_36"] - 100 / 1600 * 36) < 1e-9           # per 36 from Base Totals minutes

        # nba_api-shaped columns flow through feature engineering (tracking/hustle/synergy kept)
        import features as F
        rng = np.random.default_rng(0)
        big = pd.concat([abt.assign(player_id=abt.player_id + 10 * i) for i in range(40)], ignore_index=True)
        num = [c for c in big.columns if big[c].dtype.kind == "f" and not c.startswith("w_")]
        big[num] = big[num] * rng.uniform(0.7, 1.3, (len(big), len(num)))
        _, X, feats, _, _ = F.build_features(big, lambda *_: None)
        assert {"drives_36", "deflections_36"} <= set(feats) and any(f.startswith("pt_") for f in feats)
        assert np.isfinite(X).all()
    finally:
        for m, cls, orig in patched:
            setattr(m, cls, orig)
        ingest.time.sleep = orig_sleep


def write_fake_cache(cache: Path, seasons, n_players=120, empty_first=("synergy", "hustle"), seed=1):
    """Raw-dict cache for many players with random values, written where fetch_nba_api would write it."""
    rng = np.random.default_rng(seed)
    zones = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range", "Left Corner 3", "Right Corner 3",
             "Above the Break 3", "Backcourt"]
    for si, s in enumerate(seasons):
        ids = np.arange(1, n_players + 1) + 10 * si                       # partial overlap across seasons
        gp, mins = rng.integers(25, 82, n_players), rng.uniform(600, 2800, n_players)
        for group, _, mod, cls, kw in ingest.build_call_plan([s]):
            if si == 0 and group.split("_")[0] in empty_first:
                d = {"resultSets": [{"name": cls, "headers": ["PLAYER_ID"], "rowSet": []}]}
            elif group == "shotloc":
                base = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE"] + ["FGM", "FGA", "FG_PCT"] * 7
                rows = [[int(i), f"Fake {i}", 1, "AAA", 25] + [v for _ in zones for v in (3, int(rng.integers(0, 200)), 0.4)]
                        for i in ids]
                d = {"resultSets": {"name": "ShotLocations", "rowSet": rows, "headers": [
                    {"name": "SHOT_CATEGORY", "columnSpan": 3, "columnsToSkip": 5, "columnNames": zones},
                    {"name": "columns", "columnSpan": 1, "columnNames": base}]}}
            elif group.startswith("synergy"):
                pick = ids[rng.random(n_players) < 0.6]
                d = _rs(cls, ["PLAYER_ID", "PLAYER_NAME", "POSS"],
                        [[int(i), f"Fake {i}", float(rng.integers(5, 300))] for i in pick])
            else:
                cols = list(ingest.NBA_COLUMN_MAP.get(group, {}).values())
                if group == "dash_Base_Totals":
                    cols = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PTS"]
                h = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP", "MIN"] + cols
                rows = []
                for j, i in enumerate(ids):
                    v = rng.uniform(1, 100, len(cols)).round(3).tolist()
                    if group == "dash_Base_Totals":
                        fga = rng.uniform(150, 1500)
                        f3a = fga * rng.uniform(0, 0.6)
                        v = [fga * 0.45, fga, f3a * 0.35, f3a, fga * 0.2, fga * 0.25, fga * 1.1]
                    rows.append([int(i), f"Fake {i}", "AAA", 25, int(gp[j]), float(mins[j])] + v)
                d = _rs(cls, h, rows)
            f = ingest._cache_file(cache, group, s, "Regular Season")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(d))


def test_nba_api_end_to_end():
    """The exact user command `run_pipeline --source nba_api --no-fetch` on a fake cache (scratch dir).

    The first season has no Synergy/hustle: it must be dropped so those features survive."""
    import run_pipeline
    scratch = Path(tempfile.mkdtemp(prefix="archetype_nba_scratch_"))
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24"]
    write_fake_cache(scratch / "data" / "raw" / "nba_api", seasons)
    pa, summ = run_pipeline.run("nba_api", scratch, ["--no-fetch", "--seasons", seasons[0], seasons[-1]],
                                k_range=range(4, 6), n_boot=2)
    assert sorted(pa.season.unique()) == seasons[1:]                  # partial season dropped
    feats = json.loads((scratch / "models" / "nba_api" / "features.json").read_text())["features"]
    assert any(f.startswith("pt_") for f in feats) and "deflections_36" in feats, feats
    assert pa.player_id.dtype.kind == "i"                             # NBA PLAYER_IDs stay integers
    assert (scratch / "outputs" / "nba_api" / "report.md").read_text().startswith("# NBA player archetypes (nba_api)")
    shutil.rmtree(scratch)


def test_coverage_constant_column():
    """Regression (first live run): BOX_OUTS came back all-zero in 2016-17 and crashed PCA with NaN.

    Seasons missing a whole group are dropped; a lone absent/constant feature is dropped instead of a season."""
    rng = np.random.default_rng(0)
    rows = []
    for s in ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20"]:
        for i in range(50):
            r = {"player_id": i, "season": s, "usg": rng.normal(), "height": rng.normal(),
                 "deflections_36": rng.normal(), "contested_36": rng.normal(), "screen_ast_36": rng.normal(),
                 "box_outs_36": rng.normal()}
            if s == "2014-15":
                r.update(deflections_36=np.nan, contested_36=np.nan, screen_ast_36=np.nan, box_outs_36=np.nan)
            if s == "2016-17":
                r["box_outs_36"] = 0.0                                  # present but a constant placeholder
            rows.append(r)
    out = ingest.restrict_to_full_coverage(pd.DataFrame(rows))
    assert "2014-15" not in set(out.season) and "2016-17" in set(out.season)
    assert "box_outs_36" not in out and "deflections_36" in out
    import features as F                                                # safety net in features.py
    df = pd.DataFrame(rows).query("season != '2014-15'")
    _, X, feats, _, _ = F.build_features(df, lambda *_: None)
    assert "box_outs_36" not in feats and np.isfinite(X).all()


def test_kaggle_schema():
    """Real-data checks (skipped if the Kaggle export is not downloaded)."""
    raw = ROOT / "data" / "raw"
    if not (raw / "kaggle_bref" / "Advanced.csv").exists():
        print("  (skip kaggle schema: data not downloaded)")
        return
    df = ingest.load_kaggle(raw, 2024, 2024)
    # traded players can exceed 82 games (Bridges 83 in 2022-23, Hield 84 in 2023-24)
    assert not df.duplicated(["player_id", "season"]).any() and df.g.max() <= 85
    j = df[df.player_id == "jokicni01"].iloc[0]
    assert (j.per, j.ws_48, j.bpm, j.vorp) == (31.0, 0.299, 13.2, 10.6), (j.per, j.ws_48, j.bpm, j.vorp)
    assert df.mp.max() < 3500 and df.height.between(66, 92).all()


def test_crosswalk():
    """nba_api-style rows -> B-Ref profile columns via (season_end, normalised name); GP mismatch blanks."""
    raw = ROOT / "data" / "raw"
    if not (raw / "kaggle_bref" / "Advanced.csv").exists():
        print("  (skip crosswalk: data not downloaded)")
        return
    nba = pd.DataFrame({"player_id": [203999, 1, 2], "player_name": ["Nikola Jokic", "Nobody Atall", "LeBron James"],
                        "season": "2023-24", "season_end": 2024, "g": [79, 50, 30]})
    out = ingest.attach_bref_profile(nba, raw).set_index("player_id")
    assert out.loc[203999, "bref_id"] == "jokicni01" and out.loc[203999, "bpm"] == 13.2
    assert pd.isna(out.loc[1, "bpm"])                       # unmatched name stays NaN
    assert pd.isna(out.loc[2, "bpm"])                       # LeBron played 71, not 30 -> blanked


def test_shrinkage():
    import features as F
    df = pd.DataFrame({"season": ["s"] * 3, "fg3_pct": [1.0, np.nan, 0.3], "w_fg3_pct": [1, 0, 900]})
    out = F.shrink_rates(df, lambda *_: None)
    m = (1 + 0.3 * 900) / 901
    assert abs(out.fg3_pct[1] - m) < 1e-9 and out.fg3_pct[0] < 0.4         # 1-for-1 shrunk hard to the mean


def test_end_to_end_synthetic():
    """Full pipeline on labelled synthetic data in a scratch copy; real outputs must be untouched."""
    import run_pipeline
    real = ROOT / "outputs" / "kaggle" / "report.md"
    before = real.stat().st_mtime if real.exists() else None
    scratch = Path(tempfile.mkdtemp(prefix="archetype_scratch_"))
    pa, summ = run_pipeline.run("synthetic", scratch, k_range=range(4, 9), n_boot=3)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(pa.true_archetype, pa.archetype_id)
    assert ari > 0.5, ari
    rep = (scratch / "outputs" / "synthetic" / "report.md").read_text()
    assert rep.startswith("> **SYNTHETIC DATA") and "Synthetic Player" in rep
    feats = json.loads((scratch / "models" / "synthetic" / "features.json").read_text())["features"]
    assert not set(feats) & set(C.PROFILE_ONLY)
    for png in ["k_selection", "centroid_heatmap", "archetype_profiles", "pca_scatter", "transitions"]:
        assert (scratch / "outputs" / "synthetic" / f"{png}.png").stat().st_size > 10_000
    assert (real.stat().st_mtime if real.exists() else None) == before
    print(f"  synthetic recovery ARI={ari:.3f}, k={pa.archetype_id.nunique()}")
    shutil.rmtree(scratch)


if __name__ == "__main__":
    for t in [test_season_mapping, test_norm_name, test_shrinkage, test_nba_api_path, test_nba_api_end_to_end, test_coverage_constant_column, test_kaggle_schema, test_crosswalk,
              test_end_to_end_synthetic]:
        t()
        print(f"ok {t.__name__}")
    print("all smoke tests passed")
