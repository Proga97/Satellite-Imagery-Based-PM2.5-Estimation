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
