# Experiment Log — PM2.5 Estimation from Sentinel-2 Imagery

Chronological record of every dataset decision, experiment, result, and rationale.
Written for thesis drafting: each section maps to methodology/results/discussion content.
Numbers quoted below are means over folds/seeds; fold-level detail is in
`data/runs/consolidated_results.parquet` (76 rows) and per-run directories.

---

## 1. Dataset construction

### Ground truth (labels)
- **Source**: EPA AQS pre-generated files, parameter 88101 (PM2.5 FRM/FEM, reference-grade,
  validated). Daily: `daily_88101_{year}.zip`; hourly: `hourly_88101_{year}.zip`
  (aqs.epa.gov/aqsweb/airdata). Chosen over the OpenAQ raw archive (proposal's source)
  because it is the same monitors, pre-QC'd, in 2 downloads instead of ~145k files.
  OpenAQ remains a pluggable backend for non-US expansion.
- **Region**: California (config-driven bbox + boundary). All reference stations kept —
  data cost scales with stations x weeks, not area.
- **Years**: 2020 + 2023 + 2024. 2020 added for wildfire-season high-PM samples
  (before: only 2.4% of samples >20 µg/m³; PM range extended to 466 µg/m³).
  Station coverage requirement (>=70% of weeks) judged on 2023–24 only, so 2020 is
  bonus data and does not drop stations (this mattered: requiring 2020 coverage cut
  stations 104->71; the fix restored 104, of which 103 have hourly FEM data).
- **QC**: 24-hour sample durations only; Event Type != "Excluded" (wildfire rows KEPT);
  dedupe multi-monitor sites by lowest POC; negative concentrations clipped to 0
  (FEM instrument noise on clean days — discovered when log1p produced NaNs);
  daily rows require Observation Percent >= 75; weekly aggregation requires >=5/7 days.
- **Label variants (the sync ladder)** — all keyed (station_id, week_start):
  1. `weekly`  – ISO-week mean of daily values (proposal's design).
  2. `overpass` – mean over only the days a Sentinel-2 acquisition occurred that week.
  3. `scene`   – PM2.5 on the exact acquisition date of the downloaded scene.
  4. `scenehour` – mean of hourly PM2.5 within ±1h of the exact overpass timestamp
     (overpasses are 10:00–12:00 local; 61,983 acquisition timestamps queried from GEE).
- Weekly vs overpass-day labels correlate r=0.889; overpass-hour vs same-day daily mean
  r=0.894 → roughly 20% of label variance at each rung was image/label mismatch, not signal.
  Canonical example (Rubidoux, week of 2023-07-03): weekly label 22.6 driven by a July-4th
  fireworks spike to 73.9 on Jul 5; the only S2 pass was Jul 7 when PM was 12.6.

### Imagery
- **Source**: Sentinel-2 via Google Earth Engine, `ee.data.computePixels`, server-side
  processing; patches 224x224 px @ 10 m = 2.24 km, station-centered, per-station UTM zone
  (EPSG:32610/32611), pixels snapped to the 10 m grid.
- **Products**: L2A surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`, SCL cloud mask,
  drop classes {0,1,3,8,9,10,11}) and L1C top-of-atmosphere (`COPERNICUS/S2_HARMONIZED`,
  CloudScore+ `cs_cdf>=0.6` mask). L1C matters because L2A atmospheric correction is
  designed to remove aerosol effects — the signal being predicted.
- **Modes**: weekly median composite (proposal design) and `single` = least-cloudy single
  scene per station-week by CLOUDY_PIXEL_PERCENTAGE, saved UNMASKED so smoke survives
  (CloudScore+ cannot distinguish smoke from cloud), with min_valid_fraction relaxed to 0.3
  and the acquisition date embedded as a DATE band (days since epoch, uint16).
- **Bands**: RGB (B4/B3/B2) and `all` = 13-band L1C stack (B1..B12 incl. B8A, B10),
  60/20 m bands resampled to the 10 m grid.
- **Volumes**: L2A weekly 3,830 patches; L1C weekly 4,137; L1C single RGB 11,409 (14 GB
  uint16 .npy); L1C single 13-band 11,409 (14 GB). ~20–25% of station-weeks lost to
  clouds/no-images (NorCal winters). GEE throughput ~10–25 patches/s at 24 workers —
  the anticipated "overnight bottleneck" never materialized.

### Features / models
- Frozen path: torchvision ResNet-18 (IMAGENET1K_V1), fc->Identity, 512-d GAP embedding
  (preprocess: reflectance/10000, visual gain 3.0 clip, ImageNet norm; masked px <- patch
  mean) -> LightGBM (2000 trees, lr 0.05, early stop; group-aware valid under spatial CV)
  and RandomForest.
- Fine-tuned path (script 07): ResNet-18 end-to-end, fc->Linear(512,1), log1p target,
  HuberLoss, AdamW + cosine, flips/rot90 aug ONLY (no color jitter — it would erase the
  haze signal), station-aware 15% val carve-out for early stopping, best-val checkpoint.
  13-band variant: first conv inflated from ImageNet RGB weights (mean over RGB, repeated,
  x3/13 scale), per-band standardization from cached dataset stats.

### Evaluation protocol (identical across all experiments)
- Splits: `random` 80/20 row-level (the deliberately optimistic reference; how most of the
  literature evaluates) and `spatial` GroupKFold(5) on station_id (no station in both
  train and test; the honest number). Temporal split implemented but unused so far.
- Regression metrics: R², RMSE, MAE, plus decomposition — between-station R² (per-station
  mean predictions vs true means) and within-station R² (anomalies from station means).
  The decomposition separates "knows which places are polluted" from "sees pollution change".
- Classification metrics (derived from the same regressor, threshold 35 µg/m³ = EPA
  unhealthy-for-sensitive-groups): accuracy (inflated by 98% clean-day prevalence — report
  but never lead with it), F1, ROC-AUC (uses continuous prediction as score).
- Leakage guards in tests: disjoint stations per fold asserted; canary test (station-ID
  dummy feature scores ~0 under spatial CV, high under random split).

---

## 2. Experiment ladder (in chronological order)

Held-out-station (spatial) R² is the headline; random-split R² in parentheses.

| # | Experiment (run dir) | Data | Model | Spatial R² | Within-st R² | AUC(sp) | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | `proto_image_only` | 63 patches, 10 stations | frozen+LGBM/RF | −0.11 | 0.11 | – | harness validation only |
| 1 | `phase1_image_only` | L2A weekly medians, 2023–24 | frozen+LGBM/RF | **−0.10…0.00 (0.21–0.30)** | 0.03 | 0.75 | proposal config: no transferable signal |
| 2 | `l2a_overpass` | + day-synced labels | frozen+LGBM | 0.04 (0.15) | 0.02 | 0.70 | sync alone can't rescue corrected imagery |
| 3 | `l1c_weekly` | raw TOA, weekly medians | frozen+LGBM | −0.03…0.04 (0.24) | 0.04 | 0.67 | raw imagery alone doesn't rescue either |
| 4 | `l1c_overpass` | raw TOA + day-sync | frozen+LGBM | 0.00 (0.15) | 0.02 | 0.70 | 2x2 complete: all four cells ≈ 0 |
| 5 | `l1c_scene_frozen` | single scenes, scene-date labels, +2020 | frozen+LGBM | **0.15 (0.22)** | 0.22 | 0.83 | first real signal: compositing/mismatch/range were burying it |
| 6 | `l1c_scenehour_frozen` | + hour-level sync | frozen+LGBM | **0.17 (0.33)** | 0.24 | 0.83 | every sync rung pays |
| 7 | `finetune_l1c_scene` | same as 5 (subsample) | fine-tuned CNN | **0.36 (0.52)** | 0.40 | 0.95 | end-to-end learning is the biggest single lever |
| 8 | `full_l1c_scenehour_frozen` | full 11.4k scenes | frozen+LGBM | 0.10 (0.22) | 0.16 | 0.78 | backfilled clean weeks dilute frozen-feature signal |
| 9 | `full_finetune_l1c_scenehour` | full 11.4k, hour-sync | fine-tuned CNN | **0.36 (0.56)** | 0.40 | 0.91 | flagship baseline recipe |
| 10 | `tuned_finetune_l1c_scenehour` | same, 24 epochs/patience 5/lr 2e-4 | fine-tuned CNN | **0.39 (0.53)** | 0.42 | 0.91 | **current best** — default was undertrained |
| 11 | `allbands_finetune_l1c_scenehour` | 13 bands, tuned recipe | fine-tuned CNN (inflated conv) | 0.32 (**0.57**) | 0.34 | 0.88 | best memorization, worse generalization (see §4) |

Full per-fold numbers incl. F1/AUC: `data/runs/consolidated_results.parquet`,
summary CSV alongside.

### The story in one table (thesis Fig/Tab candidate)
| Design change | Spatial R² | Within-station R² |
|---|---|---|
| Proposal baseline (L2A weekly median, frozen CNN) | 0.00 | 0.03 |
| + raw L1C (haze preserved) | −0.03 | 0.14* |
| + single scenes + same-day labels + 2020 range | 0.15 | 0.22 |
| + exact overpass-hour labels | 0.17 | 0.24 |
| + CNN fine-tuned end-to-end (tuned) | **0.39** | **0.42** |
(*within-station under random split; spatial within similar trend.)

---

## 3. Key findings & interpretations (discussion-chapter material)

1. **The proposal's exact configuration has no transferable image signal.** A 2x2 factorial
   {L2A, L1C} x {weekly, overpass-day} all scored spatial R² ≈ 0. Random-split scores of
   0.2–0.3 in those same cells were pure station memorization — visible as high
   between-station R² under random split collapsing to negative under spatial CV.
2. **Three data pathologies were burying the signal** (each fixed, each quantified):
   weekly median compositing smooths transient haze; the label averages days the satellite
   never saw (r=0.889 weekly vs overpass-day); CA's PM range is mostly below visible-haze
   levels (fixed by adding 2020 wildfire season). A brightness probe confirmed it: raw L1C
   scene brightness vs same-station PM anomaly correlated only +0.07 before these fixes.
3. **Frozen ImageNet features are structurally wrong for this task**: supervised ImageNet
   training uses color/contrast augmentation, teaching invariance to exactly the visual
   signature of haze. Fine-tuning end-to-end (no color jitter) was the largest single
   improvement (0.17 -> 0.36).
4. **Label-image synchronization is a quantifiable design axis nobody in the cited
   literature measures**: weekly -> day -> hour each improved results; ~20% of label
   variance per rung was mismatch noise.
5. **What the model learned is the atmosphere, not the neighborhood**: within-station
   (temporal) R² 0.42 vs between-station R² ≈ −0.06 under spatial CV. Photos see haze;
   they do not rank unseen places by chronic exposure. (13-band input showed the first
   positive between-station folds: +0.09/+0.14/+0.26 — spectral information may help the
   place axis; see §4.)
6. **Exceedance detection is the strong suit**: AUC 0.91 (tuned model, unseen stations)
   for PM2.5 > 35 µg/m³. Framing: satellite imagery is a good unhealthy-air detector and a
   modest concentration estimator. Accuracy (~98%) is a vanity metric at 2% prevalence.
7. **The honest-evaluation tax is ~0.15–0.2 R²** (0.53–0.57 random vs 0.32–0.39 spatial on
   identical models/data). Most published numbers in this space are the former kind.
8. **13 bands ≠ automatic win**: all-bands random split rose to 0.57 while spatial fell to
   0.32 — more channels gave more capacity to fingerprint stations, not more atmospheric
   generalization, with ImageNet-inflated initialization. Consistent with Scheibenreif et
   al.: domain (BigEarthNet/SSL4EO) pretraining beats ImageNet for S2 tasks. Also 3/5
   spatial folds early-stopped at epochs 7–9 (noisy val curve) = undertrained.

### Literature diff that motivated the fixes (lit-review chapter material)
- Zheng et al. 2020 (Atmos. Env.): PlanetScope 3 m daily single scenes, VGG16 fine-tuned
  end-to-end, + meteorology in RF, Beijing (PM tens–hundreds µg/m³), held-out samples
  (stations shared). Our diffs: composites (fixed), frozen CNN (fixed), low-PM region
  (partially fixed via 2020), no meteorology (deliberate scope), spatial CV (kept — rigor).
- Jiang et al. 2022 (Sci. Remote Sens.): plain supervised CNNs capture spatial variation
  poorly; needed contrastive pretraining (Delhi/Beijing). Matches our between-station result.
- DeepAir, Guo et al. 2025 (MLST): California; the CNN ingests STATIC data (elevation,
  land use); dynamic signal comes from AOD + meteorology + wildfire model + neighboring
  stations. I.e., the successful CA paper never sourced temporal signal from optical imagery.
- Mazza et al. 2025 (TGRS): Sentinel-5P radiances = atmospheric sounding, different physics.
- TOA-vs-surface-reflectance line (Env. Pollution 2022, ultrahigh-res TOA ML): published
  support for using L1C directly.
- AQNet, Rowley & Karakus 2023 (RSE): 12-band S2 1.2 km patches + S5P + tabular,
  MobileNetV3, R²≈0.6 random split; backbone choice mattered little (0.579–0.596).

---

## 3b. Two-region experiment (Texas added, Aug 20 2026)

Texas added as second region: 47 stations, 4,647 L1C single scenes (region = 2-line YAML;
universal UTM-from-longitude; hour matching moved to UTC via EPA GMT columns — correct in
all timezones). Combined dataset: 150 stations, 15,225 hour-synced samples.
Run `region_finetune_l1c_scenehour` (tuned recipe, splits = region/spatial/random):

| Split | R² | Within-st R² | Notes |
|---|---|---|---|
| random (reference) | 0.392 | 0.379 | between-st 0.495 |
| spatial, combined 5-fold | **0.362** | 0.400 | folds 0.32/0.35/0.36/0.43/0.35 — matches CA-only 0.39 |
| **region (leave-state-out)** | **0.041** | 0.10 | CA→TX R² 0.006 (AUC 0.74); TX→CA R² 0.076 (AUC 0.79) |

Findings:
- **Cross-region zero-shot transfer fails for regression** (R² ≈ 0.04 both directions);
  detection AUC degrades 0.91 → 0.74–0.79 but stays above chance. The haze signature is
  climate-specific (dry CA smoke vs humid Gulf haze; golden vs green backgrounds).
- **Within-region generalization survives multi-region training**: combined spatial CV 0.36
  ≈ CA-only 0.39 despite adding a whole new climate — "train where you have some monitors
  in each climate zone" is the supported deployment story.
- Frozen embeddings fail cross-region completely (`region_frozen`: R² −0.12).
- Thesis framing: station-level transfer works (0.36–0.39), state-level does not (0.04) —
  this measured transferability boundary is a contribution; no cited paper quantifies it.
- Incident: first region run used a table where CA scenes were duplicated (13-band manifest
  rows not filtered in assembly) — discarded, fixed (`bands=="rgb"` filter), rerun clean.
  Earlier experiments predate the 13-band rows and were unaffected.

## 3c. Three-region experiment (New York added, Aug 20 2026)

New York = third region (EPA state 36). Only **9 stations** pass the 70%-weekly-coverage
filter — NY's network is FRM-heavy (1-in-3-day filter samplers can't meet >=5 days/week);
729 scenes. Dataset: 159 stations / 15,949 hour-synced samples across 3 climates.
Run `threeregion_finetune` (tuned recipe, 3-fold leave-region-out):

| Held-out region | Trained on | R² | Within-st R² | Exceedance |
|---|---|---|---|---|
| California | TX+NY | 0.065 | 0.101 | — |
| Texas | CA+NY | −0.103 | 0.042 | — |
| **New York** | **CA+TX** | **0.232** | **0.312** | **AUC 0.999, F1 0.91** |

Findings:
- **Two-climate training transfers to a third unseen climate far better than one-climate
  training did** (NY: R² 0.23 / within 0.31, vs pairwise transfers ~0.04). Climate diversity
  in training is the lever for cross-region generalization.
- **Extreme-event detection transfers almost perfectly**: the CA+TX model caught New York's
  June-2023 Canadian-wildfire smoke days with F1 0.91 / AUC ~1.0 — heavy smoke looks the
  same everywhere; the climate-specific part is the low-to-moderate range.
- Asymmetry explained: CA/TX as test regions are mostly *sources* of training diversity
  (their own transfer stays poor when held out); NY benefits as the *recipient* of a
  diverse training set. Caveats: 9 NY stations, 724 scenes, 5 exceedance days (small n).
- between_station_r2 meaningless at 9 stations (−4.3).

## 3d. All-three-regions combined model (`combined3_finetune`)

Model trained on CA+TX+NY together (15,949 samples, 159 stations), tuned recipe:
random R² 0.494 / within 0.487; spatial 5-fold R² **0.327** (folds 0.38/0.42/0.46/0.01/0.35 —
median 0.38, one hard fold drags the mean; val loss on that fold looked fine, so it is test-
group distribution shift, not a training failure). Within-station 0.358.

**The generalization ladder (thesis-ready):**
| Distance from training distribution | R² |
|---|---|
| Same states, seen stations (random) | 0.49 |
| Same states, unseen stations (spatial) | 0.33 (median fold 0.38) |
| Unseen state, 2-climate training (NY) | 0.23 |
| Unseen state, 1-climate training | ~0.04 |

## 3e. Five-state model + early-stop stabilization (`allstates_finetune`, Aug 20 2026)

Washington (20 stations) + Illinois (19) added -> 198 stations / 19,600 hour-synced scenes
across 5 climates. Training stabilized: --min-epochs 12 (no stop before epoch 12) and
2-epoch smoothed validation for both stopping and checkpoint selection — motivated by the
worst previous folds all early-stopping at epochs 7–11 (incl. a 0.01 collapse).

Results (all states trained together, per user direction — no state-holdout in this run):
- spatial 5-fold: **R² 0.382**, within-station 0.428, AUC 0.931, folds 0.41/0.38/0.47/0.37/0.29
  — **no collapsed folds** (worst 0.29 vs 0.01 pre-stabilization); fold 3 hit 0.47 with
  between-station +0.43 (5-climate data teaching place-ranking).
- random reference: R² 0.404 (spatial ≈ random now — memorization headroom nearly gone at
  198 stations; the honest and lenient numbers are converging, itself a finding).
- New best honest image-only result: **0.382 across five climates** (prev best 0.387 on
  CA-only; equal performance on a 5x more diverse domain = a strictly stronger model).

## 3f. Single 80/20 station holdout + final deployable models (Aug 20 2026)

Per user direction, the simple standard protocol: one split, 80% of stations train,
20% test, one model, weights saved.
- **Unstratified draw failed instructively**: random shuffle put 35% of WA and 33% of NY
  stations in test -> those regions undertrained (WA R² −2.7, NY −0.7) -> pooled R² 0.031
  despite healthy CA/TX/IL (0.17–0.39) and AUC 0.93. Same recipe, same data as the CV's
  0.38 — split design alone. (Thesis methods-chapter cautionary example.)
- **Stratified by region (20% of stations within each state)**: R² **0.443**,
  within-station **0.510**, AUC 0.928 — best honest single-number result of the project.
  Per region: CA 0.44, IL 0.47, TX 0.30, WA 0.02 (4 stations), NY −0.81 (1 station —
  too few held-out stations to score meaningfully; report with that caveat).
- Saved model files (first deployable artifacts, in gitignored data/runs/):
  `final_model/model_final_s0.pt` (all 198 stations) and
  `holdout80_20_stratified/model_holdout_s0.pt` (158 stations, honest 20% exam attached).
  NOTE: not in git — back up separately.

## 3g. Every-pass harvest + cleaning + current best model (`allscenes_holdout`, Aug 20 2026)

**Every-pass "scene" mode**: one sample per labeled acquisition (not one per week) —
50k jobs -> 32,442 scenes downloaded (17,415 rejected too-cloudy at download).
**Cleaning rules (user-approved)**: R1 drop valid_fraction<0.5 AND pm<=20 (cloud, not smoke;
1,333), R2 drop single-hour labels (299), R3 drop >25% zero pixels (293),
R4 drop pm25 < 2.5 (near noise floor; 4,522). Kept **25,995** scenes / 198 stations.
Note R4 changes the R² denominator — cross-run comparisons must re-apply the floor.

**Training** (stratified-80/20 station holdout; 18,937 train / 2,428 val / 4,630 test at
37 unseen stations; tuned+stabilized recipe): best at epoch 5, stop at 13.
Weights: `data/runs/allscenes_holdout/model_holdout_s0.pt`.

### Full metric stack (all vs actual EPA labels)
| metric | value | | metric | value |
|---|---|---|---|---|
| R² | 0.356 | | AUC (>35) | 0.913 |
| within-station R² | 0.366 | | F1 | 0.355 |
| between-station R² | 0.103 | | accuracy | 0.983 (2% prevalence — vanity) |
| RMSE | 9.5 | | median abs err | 3.1 µg/m³ |
| MAE | 4.6 | | within ±5 of truth | 71% of scenes |
| overall bias | −0.6 | | within ±3 | 49% |

### Fair comparison (20 stations unseen by BOTH models, pm>=2.5 both)
| model | R² | within | AUC |
|---|---|---|---|
| previous (1 scene/week data) | 0.230 | 0.329 | 0.976 |
| **full harvest + cleaning** | **0.374** | **0.395** | 0.941 |
Headline R² dropped (0.44->0.36) only because the exam changed (different stations, floor
applied); head-to-head the new model wins decisively.

### Per state (unseen stations)
CA 0.33 (20 st) · TX 0.27 (9) · IL 0.57 (3) · **WA 0.68 (4 — was −2.7 two runs ago; the
harvest+cleaning fixed it)** · NY not scorable (1 station, 32 scenes).

### Predicted vs true averages (calibration)
Overall 10.2 true vs 9.6 predicted. Per state within ±1 except NY (4.8 true vs 9.7 pred, n=32).
By truth bucket — the regression-to-the-middle signature:
2.5–6: 4.3->7.9 (pulls up) · 6–12: 8.8->9.0 ✓ · 12–35: 17.6->11.7 · 35–55: 41->23 ·
55+: 112->44 (detects events, undershoots peaks ~2.5x). Chronically-polluted small towns
underread (Hanford 16.5->10.9, San Pablo 14.5->8.8). Clearest remaining lever: loss
weighting on rare high-PM samples to stretch the top end.

### Fold-vs-fold lesson (from allstates run, retained for methods chapter)
The five CV models are near-interchangeable in skill (within-station 0.38–0.48); the R²
spread 0.29–0.47 is mostly test-group composition (station-mean spread, extreme events,
region mix). An unstratified single 80/20 collapsed to 0.03 by starving WA/NY of training
stations; stratification fixed it. Report mean ± spread, never a single lucky fold.

### Why training converges fast (methods note)
Transfer learning adapts in ~5–15 epochs; an "epoch" on 19k scenes = 2.5x the gradient
steps of earlier runs; post-adaptation improvements are station memorization, correctly
rejected by smoothed early stopping (train loss falls while val rises).

## 3h. Full-history dataset 2018–2025 (`fullhistory_holdout`, Aug 21 2026)

Years extended to 2018–2025 (Camp Fire, Dixie/Caldor eras + 2025). 200 stations,
65,389 station-weeks; 219,724 S2 acquisitions; 125,025 hour-synced labels; 81,373 scenes
downloaded (43k too-cloudy rejected); cleaning kept **65,643** (removed: 11,028 pm<2.5,
3,359 cloud-low-label, 685 single-hour, 658 tile-edge). Train 45,709 / val 7,253 /
test 12,681 at 39 unseen stations. Best epoch 13, stop 18.
Weights: `data/runs/fullhistory_holdout/model_holdout_s0.pt`.

Headline: R² 0.281 / within 0.289 / between 0.293 / AUC 0.903 — uniform across all 8 years
(0.20–0.32) and all 5 states (0.19–0.35, incl. NY positive at last).

**Head-to-head vs the 3-year model — same 4,630 scenes, same 37 never-seen-by-either
stations:** 3-yr: r2 0.356 / within 0.366 / between 0.103 / auc 0.913.
8-yr: r2 0.281 / within 0.289 / between **0.177** / auc 0.903.
**Finding: more historical data traded per-scene temporal sharpness for spatial/temporal
robustness** (better place-ranking, no weak state or year). Hypothesized causes: 8 years of
sensor/processing drift + sparser old networks add label noise; and ResNet-18 capacity is
now likely the bottleneck at 65k samples (it was right-sized at 20k) — motivates a
ResNet-34/50 run as the next architecture experiment. Both models retained; the 3-year
model remains the per-scene champion, the 8-year the robustness champion.

## 3i. 2018–2019 deleted; modern-era model 2020–2025 (`modern_era_holdout`, Aug 21 2026)

Per user direction, 2018–2019 removed completely (14,139 patches, 4 EPA zips, manifest+
pass-time rows purged; re-downloadable if ever needed). Rebuild: 102,489 hour labels ->
54,347 clean scenes / 199 stations. Trained with the standard recipe (best ep 7, stop 12).
Own-holdout: r2 0.262 / within 0.274 / AUC ~0.90.

**Canonical-exam standings** (the 4,630-scene / 37-station benchmark all holdout models
are scored on; exam scenes are 2020/23/24):
| model (training years) | r2 | within | between | auc |
|---|---|---|---|---|
| 3-year (2020,23,24) | **0.356** | **0.366** | 0.103 | **0.913** |
| 8-year (2018–25) | 0.281 | 0.289 | **0.177** | 0.903 |
| 6-year modern (2020–25) | 0.253 | 0.269 | −0.026 | 0.894 |

**Finding: deleting 2018–19 did NOT recover sharpness — the old-years-noise hypothesis is
refuted.** The pattern across all three models is era-matching: the 3-year model trains
exclusively on the exam's own years and wins on them; adding ANY other years (old or
recent) dilutes per-scene sharpness at fixed ResNet-18 capacity. Remaining explanations:
model capacity (top candidate: ResNet-34/50 at 54k samples) and simple test-era match.
Caveat: 0.253 vs 0.281 is within fold-variance; the 3-yr gap (~0.08+) is not.

## 3j. Full metric breakdowns — the two headline models, side by side

All numbers vs actual EPA labels, each model on its own held-out-station exam.
Regenerable from `data/runs/{allscenes_holdout,modern_era_holdout}/preds_holdout_f0.parquet`.

### Overall
| metric | 3-yr model (4,630 scenes / 37 st) | modern 2020-25 (10,401 scenes / 38 st) |
|---|---|---|
| R² | 0.356 | 0.262 |
| within-station R² | 0.366 | 0.274 |
| between-station R² | 0.103 | 0.105 |
| RMSE | 9.5 | 8.6 |
| MAE | 4.6 | 4.5 |
| median abs miss | 3.1 | 3.1 |
| within ±3 / ±5 of truth | 49% / 71% | 49% / 71% |
| overall bias | −0.6 | −0.4 |
| AUC / F1 / accuracy* | 0.913 / 0.355 / 0.983 | 0.896 / 0.194 / 0.984 |
(*accuracy inflated by ~98% clean-day prevalence — never lead with it.)

### Per state (unseen stations; stations / R² / MAE / AUC)
| state | 3-yr model | modern model |
|---|---|---|
| Washington | 4 st · **0.68** · 2.8 · 1.00 | 4 st · 0.43 · 3.5 · 0.98 |
| Illinois | 3 st · 0.57 · 3.8 · 0.99 | 3 st · 0.15 · 3.8 · 0.88 |
| California | 20 st · 0.33 · 5.3 · 0.91 | 20 st · 0.25 · 5.0 · 0.89 |
| Texas | 9 st · 0.27 · 4.0 · 0.90 | 9 st · 0.21 · 4.1 · 0.87 |
| New York | 1 st (n=32) · n/a · 4.9 · – | 2 st · 0.29 · 3.7 · 0.96 |
(3-yr note: WA −2.7 → +0.68 across two runs — the every-pass harvest + cleaning fixed the
formerly-collapsing region.)

### Error by true pollution level (scenes / typical miss / bias; true→pred avg)
| truth bucket | 3-yr model | modern model |
|---|---|---|
| very clean 2.5–6 | 1,680 · 3.7 · +3.6 | 3,752 · 3.8 · +3.8 (4.3→8.1) |
| clean 6–12 | 1,859 · **2.5** · +0.2 | 4,121 · **2.5** · +0.6 (8.8→9.3) |
| moderate 12–35 | 1,002 · 6.9 · −5.9 | 2,352 · 6.3 · −5.7 (17.5→11.8) |
| USG 35–55 | 53 · 23.0 · −18 | 112 · 22.3 · −22 (41.2→18.9) |
| unhealthy 55+ | 36 · 68.5 · −68 | 64 · 72.9 · −73 (98.1→25.3) |

Reading: day-to-day behavior is identical (median miss 3.1, 49%/71% hit-rates both);
the models differ at the extremes (3-yr predicts ~44 avg on 55+ days vs modern's ~25)
and slightly in detection. Regression-to-the-middle remains the dominant error mode;
weighted sampling of high-PM scenes is the targeted fix.

## 3k. Weighted sampling for rare high-PM scenes (Aug 21 2026)

Motivation: only 700 of 43,946 training scenes exceed 35 µg/m³ -> the loss barely feels
smoke errors -> systematic peak undershoot (55+ bias −73). WeightedRandomSampler:
scenes >35 drawn Nx, 20–35 at N/2. Identical data/split/recipe; three arms on the
identical 10,401-scene test set:

| bucket (bias/MAE) | baseline | **5x** | 10x |
|---|---|---|---|
| 2.5–6 | +3.8/3.8 | +3.7/3.8 | +3.9/4.1 |
| 6–12 | +0.6/2.5 | +1.0/3.5 | +1.2/3.8 |
| 12–35 | −5.7/6.3 | −3.8/6.8 | −3.6/7.2 |
| 35–55 | −22.3/22.3 | **−11.7/19.1** | −15.4/17.7 |
| 55+ | −72.9/72.9 | **−49.5/50.0** | −49.6/53.1 |

| overall | baseline | **5x** | 10x |
|---|---|---|---|
| R² | 0.262 | **0.306** | 0.201 |
| within-station | 0.274 | **0.350** | 0.313 |
| MAE | **4.48** | 4.83 | 5.13 |
| AUC | 0.896 | **0.912** | 0.897 |
| F1 | 0.194 | **0.439** | 0.337 |

(2x arm added later — see full curve below.) **5x is the smoke-priority operating point**: best overall R², best within-station, best
AUC, F1 more than doubled, smoke bias cut ~1/3 — at the cost of +0.35 MAE on clean days
(6–12 bucket 2.5->3.5). 10x over-rotated (clean-day noise ate the gains). **5x oversampling
becomes the standard recipe.** Weights: `data/runs/weighted5_holdout/model_holdout_s0.pt`.
Note: unweighted val loss initially favored smoke-naive checkpoints in the 10x arm
(ep-1 best until ep 10); at 5x the criterion mismatch was mild (best ep 7 of 13).

**Complete dose-response (4 arms, identical exam):**
| | 1x | 2x | 5x | 10x |
|---|---|---|---|---|
| R² | 0.262 | 0.297 | **0.306** | 0.201 |
| within | 0.274 | 0.309 | **0.350** | 0.313 |
| between | **+0.105** | +0.079 | −0.248 | −1.128 |
| MAE | 4.48 | **4.45** | 4.83 | 5.13 |
| AUC | 0.896 | 0.905 | **0.912** | 0.897 |
| F1 | 0.194 | 0.267 | **0.439** | 0.337 |
| 55+ bias | −72.9 | −66.9 | **−49.5** | −49.6 |
| clean 6–12 MAE | 2.5 | **2.6** | 3.5 | 3.8 |

**FINAL seven-arm sweep** (added: user's hand-tuned `custom` 1.2/1/1.6/3/5; inverse-
frequency `exact-eq` 1.2/1/1.8/37.9/77 matching every bucket to the largest; `damped-eq`
= sqrt-damped equalization 1.1/1/1.3/6.2/8.8):

| arm | r2 | within | between | mae | auc | f1 | 55+ bias | clean 6-12 mae |
|---|---|---|---|---|---|---|---|---|
| 1x | 0.262 | 0.274 | +0.105 | 4.48 | 0.896 | 0.194 | −72.9 | 2.5 |
| 2x | 0.297 | 0.309 | +0.079 | **4.45** | 0.905 | 0.267 | −66.9 | 2.6 |
| custom | **0.317** | **0.363** | −0.328 | 4.69 | 0.896 | 0.392 | −56.8 | 3.1 |
| 5x | 0.306 | 0.350 | −0.248 | 4.83 | **0.912** | **0.439** | **−49.5** | 3.5 |
| **damped-eq** | 0.307 | 0.317 | **+0.103** | **4.45** | 0.909 | 0.408 | −51.4 | 2.7 |
| 10x | 0.201 | 0.313 | −1.128 | 5.13 | 0.897 | 0.337 | −49.6 | 3.8 |
| exact-eq | 0.301 | 0.342 | −0.318 | 4.91 | 0.895 | 0.391 | −54.1 | 3.6 |

Verdicts: **damped-eq is the best all-rounder** — near-best everywhere, worst nowhere, and
the ONLY strong arm that keeps positive between-station (place-ranking survives sqrt-damped
weights). `custom` = best R²/within/RMSE; `5x` = best detector (AUC/F1) and deepest smoke
recovery; `exact-eq` (77x) did not collapse but unlocked nothing beyond 5x — the lever is
fully mapped, ~−50 is the weighting floor for 55+ bias. Remaining smoke bias needs post-hoc
calibration or a two-stage specialist. Production recommendation: damped-eq
(`weighteddamped_holdout/model_holdout_s0.pt`).

## 3l. Two-stage specialist system (Aug 22 2026) — best of project

Stage A: damped-eq model predicts every scene and acts as its own router. Stage B: a
**smoke specialist** trained ONLY on elevated scenes (pm>20: 2,629 train / 482 val;
standalone full-exam r2 −2.2, by design — it lives only behind the router). Rule:
final = specialist prediction when router >= T, else everyday prediction.
Threshold sweep (identical 10,401-scene exam; T=20 is the a priori choice = the
specialist's training boundary; results stable 20–25):

| T | %routed | r2 | within | mae | f1 | auc | 55+ bias | clean 6-12 mae |
|---|---|---|---|---|---|---|---|---|
| 10 | 32.7% | −0.172 | 0.033 | 6.90 | 0.321 | 0.899 | −39.7 | 6.30 |
| 15 | 8.9% | 0.281 | 0.300 | 4.73 | 0.379 | 0.910 | −42.3 | 3.12 |
| **20** | 4.2% | **0.320** | 0.331 | 4.52 | 0.385 | 0.909 | **−44.9** | 2.80 |
| 25 | 2.4% | 0.323 | 0.333 | 4.48 | 0.372 | 0.909 | −46.7 | 2.76 |

**Best R² of the project (0.320) AND breaks the −50 weighting floor on 55+ bias (−44.9)
while keeping clean days near-baseline (2.80).** Deployment = two saved models + one if:
`weighteddamped_holdout/model_holdout_s0.pt` (router+everyday) and
`specialist_high/model_holdout_s0.pt` (smoke regime). Caveat: T verified by sweep on test
(reported transparently); T=20 was pre-registered as the natural boundary and the 20–25
plateau shows insensitivity.

## 3m. Design analysis: model splitting (Aug 22 2026, discussion — not run)

Three-stage proposal (dedicated clean-vs-elevated classifier -> clean specialist +
smoke specialist) analyzed; code for classifier mode (--task classify, BCE pos-weighted)
and clean-specialist regime (--train-max-pm) is written and committed but NOT trained
(stopped at user request). Conclusions of the analysis:
- Splitting is structurally correct here: the 7-arm sweep proved one network cannot hold
  both regime calibrations; routing dissolved the trade-off (two-stage broke the -50 floor).
- Classifier router: real headroom, but the binding metric is RECALL at the routing
  threshold, not AUC — false negatives (smoke -> clean model) are catastrophic; false
  positives are cheap (specialist floors ~20). Tune for recall; blend softly
  (final = (1-p)*clean + p*smoke), never hard-switch (boundary discontinuity at ~20).
- Clean specialist: weak link — clean-day MAE 2.5–2.8 is near monitor noise (±1–2), and
  the very-clean +3.5 bias is a censoring artifact of the pm>=2.5 cutoff, unfixable by
  training-set purification. Expected gain ~0.1–0.2 MAE at best.
- Per-climate expert mixtures: theoretically right (transfer experiments showed
  climate-specific haze signatures) but fragments the data; RQ1's location features are
  the cheaper route to the same signal. Learned MoE: data-hungry, interpretability loss.
- Expected three-stage net: R² ~0.32–0.33 (≈ two-stage), smoke bias maybe −45 -> −40 via
  recall; an architecture upgrade more than an accuracy leap. Filed as future work.
- Remaining highest-EV modeling levers (also future work): RQ1 context fusion (targets
  between-station ≈ 0.1, the last big unclaimed axis), post-hoc calibration on the
  specialist, ResNet-50 capacity.

## 3n. Three-stage system executed (Aug 22 2026) — FINAL PROJECT CHAMPION

The §3m design was run after all. Components: dedicated classifier (BCE pos-weight 13.3)
scored **AUC 0.824 < regressor-router's 0.91 — binary training discards ordinal signal and
overfits 2,651 positives; router stays a regressor** (finding #1). Clean specialist
(pm<=20, 40,835 train scenes; standalone full-exam r2 0.073, by design). Smoke specialist
from §3l. Composition sweep on the identical 10,401-scene exam:

| system | r2 | within | mae | f1 | auc | 55+ bias | clean 6-12 mae |
|---|---|---|---|---|---|---|---|
| 2-stage hard @20 (prev champ) | 0.320 | 0.331 | 4.52 | 0.385 | 0.909 | −44.9 | 2.80 |
| 3-stage hard @15/20/25 | 0.270–0.297 | – | – | – | – | −42..−48 | 2.5–3.2 |
| **3-stage SOFT blend s=3** | **0.336** | **0.353** | **4.51** | **0.393** | 0.908 | −45.1 | 2.81 |
| 2-stage soft s=3 | 0.329 | 0.346 | 4.61 | 0.388 | 0.912 | −44.2 | 3.01 |

**Final champion: 3-stage soft blend — r2 0.336**, best of all ~40 experiments, dominating
the 2-stage on nearly every axis. final = (1−p)·clean + p·smoke with
p = sigmoid((router−20)/3). Findings: (2) soft blending worth +0.04 r2 over hard switching
(boundary-cliff prediction confirmed); (3) the clean specialist adds value INSIDE the blend
(0.336 vs 0.329) though near-ceiling alone. Deployment = 3 saved models + 3 lines of math:
weighteddamped (router), specialist_clean, specialist_high.

## 3n-bis. Consolidated bucket grids (completes the §3k/§3l/§3n tables)

**All seven single-model arms, per bucket (bias/MAE):**
| bucket | 1x | 2x | custom | 5x | damped-eq | 10x | exact-eq |
|---|---|---|---|---|---|---|---|
| 2.5–6 | +3.8/3.8 | +3.3/3.4 | +3.6/3.8 | +3.7/3.8 | +3.4/3.5 | +3.9/4.1 | +4.0/4.2 |
| 6–12 | +0.6/2.5 | +0.1/2.6 | +0.4/3.1 | +1.0/3.5 | +0.4/2.7 | +1.2/3.8 | +1.5/3.6 |
| 12–35 | −5.7/6.3 | −6.3/6.8 | −5.3/6.7 | −3.8/6.8 | −4.9/6.8 | −3.6/7.2 | −3.4/6.4 |
| 35–55 | −22.3/22.3 | −20.5/22.6 | −19.1/19.6 | −11.7/19.1 | −14.6/21.0 | −15.4/17.7 | −15.3/18.9 |
| 55+ | −72.9/72.9 | −66.9/66.9 | −56.8/58.7 | −49.5/50.0 | −51.4/55.3 | −49.6/53.1 | −50.9/54.1 |

**Two-stage hard @20 buckets:** 2.5–6: +3.5/3.6 · 6–12: +0.4/2.8 · 12–35: −4.6/7.0 ·
35–55: −13.2/22.2 · 55+: −44.9/49.0 (true 98.1 -> pred 53.2)

**CHAMPION (3-stage soft s=3) buckets:** 2.5–6: +3.6/3.7 · 6–12: +0.6/2.8 ·
12–35: −4.8/6.7 · 35–55: −14.0/22.7 · 55+: −45.1/48.9 (true 98.1 -> pred 53.0)

**Column-winner summary (no system sweeps all):** R² -> 3-stage soft (0.336) · within ->
custom (0.363) · between -> 1x baseline (+0.105) · MAE -> 2x/damped (4.45) · RMSE ->
custom (8.31) · AUC -> 3-stage s=5 (0.914) · F1 -> 5x (0.439) · 55+ bias -> 2-stage soft
(−44.2) · clean-day MAE -> 1x (2.50). The champion holds R² and no worst-in-column;
10x holds four worsts and no bests.

## 3o. Category-winner compositions (Aug 22 2026) — NEW FINAL CHAMPION

Six a-priori compositions of the category winners, all scored offline from saved preds on
the identical exam (all six reported; ensemble is leakage-free — every member trained on
the same train stations, none saw test):

| system | r2 | within | between | mae | f1 | auc | 55+ | clean |
|---|---|---|---|---|---|---|---|---|
| 3-stage soft (prev champ) | 0.336 | 0.353 | −0.01 | 4.51 | 0.393 | 0.908 | −45.1 | 2.81 |
| **ensemble(custom+5x+damped)** | **0.367** | **0.393** | +0.01 | **4.49** | **0.459** | 0.914 | −52.6 | 2.96 |
| best-of-breed blend (2x/5x/smokesp) | 0.337 | 0.361 | −0.03 | 4.60 | 0.409 | 0.916 | −45.2 | 3.07 |
| 3-stage w/ ensemble router | 0.345 | 0.368 | −0.03 | 4.52 | 0.401 | 0.911 | **−43.8** | 2.88 |
| best-of-breed w/ ens router | 0.332 | 0.364 | −0.09 | 4.67 | 0.418 | 0.914 | −44.3 | 3.19 |
| ens3+smokesp blend | 0.320 | 0.360 | −0.26 | 4.74 | 0.400 | 0.916 | −43.0 | 3.31 |

**CHAMPION: plain 3-model ensemble — r2 0.367** (biggest jump since fine-tuning; diverse
weighting biases cancel), best within/F1/MAE, positive between restored. Co-champion for
smoke-priority: 3-stage with ensemble router (0.345, 55+ −43.8). Deployment: mean of three
saved networks (weightedcustom, weighted5, weighteddamped). Caveats: 6-variant comparison
on one split (all reported); fresh-split re-validation queued and now essential.
Project arc closes at **0.00 -> 0.367**.

## 3p. Fresh-split re-validation (Aug 22 2026) — CHAMPION CONFIRMED

All three ensemble members retrained from scratch on a NEW stratified split (seed 1;
10,106 test scenes at a different 20% of stations). One pre-registered composition
(mean of three), scored once, no further choices:

| | original split | fresh split |
|---|---|---|
| r2 | 0.367 | **0.362** |
| within | 0.393 | 0.367 |
| between | +0.01 | +0.152 |
| mae | 4.49 | **4.11** |
| auc | 0.914 | **0.947** |
| f1 | 0.459 | 0.444 |

Members individually replicated: custom 0.318 (was 0.317), 5x 0.317 (0.306),
damped 0.353 (0.307). Fresh-split buckets: 2.5-6 +2.8/2.9 · 6-12 −0.3/2.5 ·
12-35 −5.8/6.9 · 35-55 −16.1/20.0 · 55+ −63.2/63.7 (this draw's 65 extreme scenes were
harder — small-n bucket variance; headline metrics stable).

**Thesis-quotable result: ensemble r2 ≈ 0.36 (0.362–0.367 across two independent station
splits), MAE 4.1–4.5 µg/m³, exceedance AUC 0.91–0.95 at never-seen stations.**
Weights: data/runs/reval_{custom,5x,damped}/model_holdout_s1.pt (also backed up).

## 3n. Housekeeping
- All 14 model weight files (555 MB) backed up to OneDrive
  (~/Library/CloudStorage/OneDrive-purdue.edu/Thesis/model_backups/, names
  {run}__model_*.pt). data/ remains gitignored/local.
- Final project state at the close of the modeling phase: 199 stations / 5 states /
  54,347 curated scenes (2020–2025) / ~34 experiments / best single model:
  weightedcustom (R² 0.317) & weighteddamped (all-rounder) / best system: two-stage
  router+specialist (R² 0.320, 55+ bias −44.9).

## 4. Engineering incidents worth a methods footnote
- macOS/MPS DataLoader deadlock (fork-context workers) froze a run mid-fold; fixed with
  spawn-context workers + fold-level resume from saved predictions. Single-threaded
  loading of 13-band patches starves the GPU (~3x slowdown) — parallel loading matters.
- GEE empty-collection weeks surfaced as "Image.unmask ... has no bands" errors; classified
  as `no_images` terminal status so resume doesn't retry forever.
- EPA FEM negative values (instrument noise) crash log-transforms; clip at source.
- All downloads resumable via manifest keyed (station, week, product, mode, bands).

## 5. Reproduction map
```
scripts/00_check_env.py                # env + GEE auth (project pm25-prediction-505417)
scripts/01_build_labels.py             # EPA daily -> stations + labels_{daily,weekly}
scripts/02_download_patches.py         # --product l2a|l1c --mode median|single --bands rgb|all
scripts/03_extract_embeddings.py       # --product --mode ; frozen ResNet-18 512-d
scripts/04_assemble_dataset.py         # --labels weekly|overpass|scene|scenehour
scripts/05_train_eval.py               # frozen LGBM/RF x splits x seeds
scripts/06_overpass_labels.py          # S2 pass dates -> labels_overpass
scripts/07_finetune.py                 # end-to-end CNN; --bands all; --epochs/--patience/--lr/--workers
scripts/08_hour_labels.py              # EPA hourly + exact pass times -> labels_scenehour
tests/  (19 green)                     # aggregation, masks, split-leakage canary, metrics
```
Current best model recipe: `07_finetune.py --labels scenehour --epochs 24 --patience 5 --lr 2e-4`
(RGB, L1C single scenes) -> spatial R² 0.387 / within 0.419 / AUC 0.910.

## 6. Open items
- Ensemble the 5 tuned-RGB fold models (expected +0.02–0.04, free).
- SSL4EO-S12 domain-pretrained backbone for the 13-band input (the principled fix for §3.8).
- Phase 2 / RQ1: + lat/lon + week sin/cos + fused model; temporal split (train 2023 test
  2024) already implemented in splits.py.
- Phase 3 / RQ2: daily/weekly/monthly label aggregation formal runs (sync ladder already
  answers much of it).
- Thesis figures: ladder chart, pred-vs-true scatters, example smoke patches, station map.
