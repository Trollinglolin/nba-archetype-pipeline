# NBA Archetype Clustering & Data Pipeline

**Live demo:** https://trollinglolin.github.io/nba-archetype-pipeline/

Discovers data-driven player archetypes (style, not quality or position) from 2013-14 to 2025-26 player-seasons
with season-relative features, PCA and K-Means. `k` is chosen by elbow, silhouette **and** a player-level bootstrap
stability test.

## What was and was not verified

| Part | Status |
|---|---|
| **Kaggle / Basketball-Reference path** (`--source kaggle`) | **Real data, run end to end here.** Downloaded 2026-11-11 (dataset files dated 2026-11-01). Results below and in `outputs/kaggle/`. |
| **`nba_api` path** (`--source nba_api`) | **Fetched live on 2026-11-12:** all 338 requests succeeded and none failed. The outputs were then rebuilt from that cache (no network). Results are in `outputs/nba_api/`. |
| Tracking / hustle column names (`DRIVES`, `CATCH_SHOOT_FGA`, `POTENTIAL_AST`, ...) | **Verified by the live fetch.** No `NOTE <group>: missing [...]` lines appeared, so every mapped column exists. |
| Synergy / hustle season coverage (live) | Synergy has data in every season from 2013-14. Hustle returns 0 rows for 2013-14 and 2014-15, and only 147 of 476 players for 2015-16, so **those 3 seasons are dropped** and 2016-17 to 2025-26 are clustered. In 2016-17 `BOX_OUTS` is present but all zeros (box-out tracking starts in 2017-18), so **`box_outs_36` is dropped and the season kept**. Rule: a season is dropped only when >= 3 features (`config.MIN_FEATS_TO_DROP_SEASON`) are absent or constant in it; otherwise those features are dropped instead. `--keep-partial-seasons` keeps every season and drops sparse features. |
| Crosswalk to B-Ref (live) | 5,491/5,492 player-seasons matched. A second pass reuses a player's B-Ref ID from his other seasons when B-Ref renames him (e.g. "Nigel Hayes" in 2018 vs "Hayes-Davis" in 2026). The one miss, Egor Dëmin 2025-26, is absent from the Kaggle export itself. |
| Full `run_pipeline --source nba_api --no-fetch` chain | Tested offline on a fake 4-season, 120-player cache in a scratch dir (`test_nba_api_end_to_end`), including a first season with no Synergy or hustle data. |
| Synthetic data | Used only in a temporary scratch directory by `tests_smoke.py`. Fictional names, and the report is headed "SYNTHETIC DATA". |

## Quick start

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -W ignore tests_smoke.py              # ~3 s, no network
.venv/bin/python -u run_pipeline.py --source kaggle    # downloads the Kaggle export if missing (11 MB), ~15 s
```

### Running the `nba_api` fetch

```bash
# 0. see the plan: 338 requests (dash 78, bio 13, tracking 91, shot locations 13, hustle 13, synergy 130)
.venv/bin/python ingest.py --source nba_api --dry-run
# 1. cheap first pass, one season (26 calls, ~1-2 min), to confirm the column names:
.venv/bin/python -u ingest.py --source nba_api --seasons 2024-25 2024-25 2>&1 | tee data/nba_api_fetch.log
# 2. everything (338 calls, roughly 14-28 min; resumable, so rerun the same command after failures):
.venv/bin/python -u ingest.py --source nba_api 2>&1 | tee -a data/nba_api_fetch.log
#    optional: --skip synergy  (208 calls)  |  --proxy http://...  |  --timeout 90
# 3. cluster from the cache (no network):
.venv/bin/python -u run_pipeline.py --source nba_api --no-fetch 2>&1 | tee data/nba_api_pipeline.log
```

After step 1, check the log for:
- `NOTE <group>: missing [...]`: an expected column name was wrong. Fix it in `ingest.NBA_COLUMN_MAP`.
- `rows share N PLAYER_IDs` in any group other than Synergy: per-team rows for traded players.
- `coverage: drop season ...`: which seasons lack Synergy or hustle.

Each response is stored untouched as `data/raw/nba_api/<group>/<season>.json`, and files that already exist are
skipped. A failed request is retried 5 times with exponential backoff, then skipped and listed at the end. The
nba_api ABT gets WS/48, BPM, VORP and position mix from the Kaggle export through a (season, normalised name)
crosswalk. Every match is validated on games played, and misses are logged so they can be added to
`ingest.NAME_ALIASES`.

### Labelling seasons outside the fitted range

`apply_frozen.py` applies the saved Kaggle model (scaler, PCA, k-means; no refit) to 1996-97 onward, using the same
per-season preprocessing. It writes `outputs/kaggle/player_archetypes_1997plus.csv`, which `../nba-award-predictor`
reads. Re-labelling 2014+ this way reproduces all 4,669 fitted labels. Note that prime Shaq and Dwight Howard come out
as Rim-Running Big: the archetype describes shot style, not offensive role.

## Pipeline

| Step | File | Key decisions |
|---|---|---|
| Ingest | `ingest.py` | One row per player-season, keyed on `player_id` + `season`. Traded players: B-Ref's combined `2TM` row is kept and the per-team rows dropped (924 combined rows; the `team` column lists the teams). Every join is asserted 1:1 and its row counts logged. Filter: g >= 20 and mp >= 500 (removes 2,273 of 6,942 rows). Writes `data/abt_<source>.csv` plus `_quality.txt` (missing values, rows per season, full log). |
| Features | `features.py` | Rate/style features only, never totals. The order matters: (1) shrink small-sample rates (3P%, FT%, assisted share, corner-3 share) toward the attempt-weighted season mean; (2) impute season medians for rare gaps; (3) winsorise at 1/99% **within season**; (4) z-score **within season**, so a 2015 and a 2025 stretch big are comparable despite 3-point drift; (5) prune pairs above \|r\| 0.9; (6) StandardScaler. |
| Model | `model.py` | PCA keeps 90% of variance (14 PCs from 29 features). K-Means over k = 4..14. **Bootstrap resamples players, not rows**, because the same player appears in up to 13 seasons. A GaussianMixture is fit as a cross-check. Clusters are named by matching centroid z-profiles to templates in `config.ARCHETYPE_TEMPLATES`. |
| Report | `report.py` | `outputs/<source>/`: `report.md`, `player_archetypes.csv`, `cluster_summary.csv`, `centroids_z.csv`, `transitions.csv`, `representatives_and_ambiguous.csv`, `sanity_players.csv`, plus 6 PNG charts. |

**Kept out of K-Means on purpose** (`config.PROFILE_ONLY`), and used only to profile clusters:
- PER, WS/48, BPM, VORP, ORtg/DRtg and on/off: these would add a "how good" axis.
- Position mix (PG%..C%): this would make the clusters reproduce positions.
- Minutes, games and age.

**Dropped by the correlation prune:** average shot distance (r = 0.93 with 3PA rate), and pts/100 and FGA/100
(r = 0.95 and 0.97 with usage).

## Results (real data, Kaggle source, 4,669 player-seasons, 1,055 players)

**k = 6** (see `outputs/kaggle/k_selection.png`):
- It is the largest k with player-bootstrap ARI >= 0.80 (0.84 ± 0.07) and every cluster >= 2% of rows.
- k = 7 reaches 0.78. From k = 9 on, stability falls to about 0.6 and lower.
- Silhouette is low everywhere (0.17 at k = 4, 0.14 at k = 6). Player styles form a continuum, not separate
  blobs, so silhouette is a weak guide here, and the elbow is gradual.
- **The GaussianMixture agrees only moderately (ARI 0.47).** The broad groups are robust, but the exact boundaries
  between neighbouring archetypes depend on the method.

| Archetype | share | MPG | USG% | height (in) | 3PA rate | WS/48 | BPM | closest to centroid |
|---|---|---|---|---|---|---|---|---|
| Movement / Spot-up Shooter | 25% | 23.5 | 16.9 | 78.2 | .56 | .083 | -0.9 | Cameron Johnson 23-24, Terrence Ross 13-14 |
| Low-Usage Role Wing | 18% | 21.1 | 16.2 | 79.0 | .38 | .074 | -1.8 | James Ennis III 19-20, Jalen McDaniels 20-21 |
| Pass-First Guard | 17% | 22.7 | 18.6 | 74.8 | .40 | .069 | -1.3 | Raymond Felton 16-17, Goran Dragić 18-19 |
| Interior Scorer (non-rim-running big) | 14% | 26.5 | 22.9 | 81.9 | .21 | .121 | +1.0 | Jon Leuer 15-16, Bobby Portis 17-18 |
| Scoring Creator | 13% | 31.6 | 27.4 | 76.2 | .35 | .114 | +2.0 | Bradley Beal 18-19, Donovan Mitchell 25-26 |
| Rim-Running Big | 13% | 20.8 | 16.1 | 82.4 | .05 | .150 | +0.2 | Ian Mahinmi 15-16, Bismack Biyombo 17-18 |

- **Not a minutes proxy.** Archetype explains 13% of the variance in total minutes and 23% in MPG (η²). It
  explains 52% for usage and 60% for height, which are style dimensions. BPM, WS/48 and VORP were never inputs;
  archetype explains about 22-24% of their variance.
- **Stability over time.** 77% of consecutive player-season pairs keep their archetype. The most common move is
  Low-Usage Role Wing → Spot-up Shooter (18%).
- **League evolution** (`archetype_share_by_season.png`):
  - Low-Usage Role Wing grew from 14% of qualified player-seasons (2013-14) to 23% (2025-26).
  - Interior Scorer shrank from 18% to 12%.
  - Scoring Creator rose from 11% to 14%.
  - These are season-relative shares, so they show how the mix of roles changed, not the league-wide rise in
    3-point volume.
- **Post-hoc sanity check** (famous players, never used to fit or name):
  - Gobert and Capela are Rim-Running Bigs every season.
  - Klay Thompson is a Spot-up Shooter every season.
  - DeRozan, LeBron and Curry are Scoring Creators.
  - Chris Paul moves from Scoring Creator to Pass-First Guard in 2022-23, as his usage fell.
  - Jokić and Giannis are Interior Scorers.
  - Draymond Green moves between Pass-First Guard, Role Wing and Interior Scorer. He is a genuinely
    position-less case.

### nba_api source (live data, 2016-17 to 2025-26, 3,617 player-seasons, 45 features incl. tracking/hustle/Synergy)

- Again **k = 6** (bootstrap ARI 0.83). GaussianMixture agreement is ARI 0.43.
- The same six archetypes reappear. The tracking features sharpen them:
  - Scoring Creators are defined by pull-ups, isolations and drives.
  - Spot-up Shooters are defined by off-screen, handoff and catch-and-shoot plays.
  - Pass-First Guards are defined by seconds per touch and touches.
- The rim-running big cluster is labeled **Rim Protector**: it ties with the Rim-Running Big template (cos 0.97),
  and its rim-defense numbers (+1.6 z in blocks and rim shots defended) make the name accurate.
- 81% of players keep their archetype from one season to the next. Archetype explains 13% of the variance in
  minutes, 53% in usage and 60% in height.
- Sanity check:
  - Giannis is a Scoring Creator here (9 of 10 seasons). He was an Interior Scorer on box-score data only, and
    drive and isolation data moves him.
  - Draymond Green groups with the bigs (Interior Scorer, 9 of 10).
  - Gobert and Capela are Rim Protectors every season.
- Players with any Synergy rows but no possessions of a play type get 0 for it. The 32 qualified player-seasons
  with no Synergy rows at all get the season median, and that is logged.

### Naming caveats

- Names come from matching each centroid's z-profile to templates written from basketball meaning. They do not
  come from players.
- After the first real run, two templates were added: **Pass-First Guard** and **Low-Usage Role Wing**. The
  first-run names, "Primary Playmaker" and "3-and-D Wing", claimed usage and defense those centroids do not show
  (steal% +0.3, block% 0.0). The clusters themselves did not change.
- At k = 6, **"Interior Scorer" is a broad non-rim-running-big group.** It holds post and face-up scorers, Jokić,
  Giannis, and stretch bigs such as Brook Lopez.
- `run_pipeline.py --k 8` splits it into low-usage stretch/rim-protector bigs (Lopez) and high-usage interior
  creators (Jokić, Giannis), and adds a Mid-Range Scorer group. But k = 8 has bootstrap ARI 0.70, below the
  stability bar, so it is offered as a finer, less reproducible view.
- Templates name about 14 archetype ideas, but only k of them are used. A cluster that matches no template
  (cosine < 0.35) is called `Mixed: <top features>`.

### Selection-rule check on synthetic data

Six archetypes were planted in the synthetic data. The "largest stable k" rule picked k = 7, one more than the
truth: k = 6 had ARI 0.996 and k = 7 had 0.887, both above the bar. Recovery was still ARI 0.91. So the rule may
split one real group in two. Compare k and k-1 in `k_selection.png` before reading too much into the smallest
archetype.

## Interpretation guide

- Every number in the heatmap and radar charts is a **within-season z-score**: +1 means one standard deviation
  above that season's qualified players. A "high 3PA rate" archetype in 2014 shot fewer threes in absolute terms
  than one in 2025.
- `player_archetypes.csv` has columns `player_id, player_name, season, team, archetype, dist_to_centroid,
  second_archetype, ambiguity`.
  - `ambiguity` = distance to the own centroid divided by distance to the next nearest one.
  - Values near 1.0 mean the player sits between two archetypes (e.g. Giannis 2018-19: Interior Scorer vs
    Rim-Running Big). Read those labels as "mostly X, partly Y".
- An archetype describes **how** a player was used, not **how well** they played. Use the WS/48, BPM and VORP
  columns in `cluster_summary.csv` for value.
- For the Kaggle source, `player_id` is the B-Ref slug (e.g. `jokicni01`). For nba_api it is the NBA `PLAYER_ID`.

## Limitations

- **Missing context.** There is no defender quality, lineup, teammate or scheme context. A spot-up shooter on a
  heliocentric team looks like the role, not the ability.
- **Box-score style features.** The Kaggle source has no tracking, play-type or hustle data. Use the nba_api
  source for those; it covers only 2016-17 onward, because of hustle coverage.
- **Tracking definitions drift.** For example, box-outs per 36 fall from about 17 to about 5 between 2017-18 and
  2025-26. Within-season z-scores absorb this, but raw tracking values are not comparable across seasons.
- **Defense is weakly measured.** Only steal%, block%, DREB%, fouls and charges are available, so defensive
  specialists cannot really be separated.
- **Small archetypes.** Rarer archetypes (e.g. post-up specialists) are absorbed into neighbours at stable k.
- **Sample filter.** Players below 20 games or 500 minutes are excluded entirely and never assigned.
- **Shortened seasons.** 2019-20 and 2020-21 are handled through rates and within-season scaling, but their
  thresholds bite slightly harder.
- **Data terms.** The data derives from Basketball-Reference via Kaggle. Check the dataset licence and
  Sports-Reference's terms before redistributing raw files. `data/` is git-ignored.

## Files

`config.py` (seasons, thresholds, feature groups, templates) · `ingest.py` · `features.py` · `model.py` ·
`report.py` · `run_pipeline.py` · `tests_smoke.py` · `requirements.txt` · `models/<source>/archetype_model.joblib`
(feature list, scaler, PCA, KMeans, names).
