# Data pipeline: Sentinel-2 scenes paired with EPA PM2.5 at the overpass hour.
# Edit configs/pipeline.yaml (regions, years, EPA parameter code) and run `make dataset`.
PY ?= python

.PHONY: check labels hour-labels patches clean-scenes context dataset train figures thesis test smoke

check:            ## verify imports, GPU/MPS, config and Earth Engine access
	$(PY) scripts/00_check_env.py

labels:           ## 1. EPA AQS daily files -> stations + daily/weekly labels (coverage filter)
	$(PY) scripts/01_build_labels.py

hour-labels:      ## 2. EPA hourly files + Sentinel-2 pass times -> label at the overpass hour
	$(PY) scripts/08_hour_labels.py

patches:          ## 3. download one 224x224 patch for every usable pass (resumable, parallel)
	$(PY) scripts/02_download_patches.py --mode scene

clean-scenes:     ## 4. quality rules: low label, cloud-vs-smoke, single-hour label, tile edge
	$(PY) scripts/09_clean_scenes.py

context:          ## 5. ERA5-Land weather, SRTM elevation, solar elevation, season per scene
	$(PY) scripts/10_context_features.py

dataset: labels hour-labels patches clean-scenes context   ## 1-6: the whole data pipeline
	$(PY) scripts/12_build_dataset.py

train:            ## one FiLM model on split A (see README for the five ensemble members)
	$(PY) scripts/07_finetune.py --experiment film_full --labels scenehour --mode scene \
	  --splits holdout --epochs 24 --patience 5 --min-epochs 12 --lr 2e-4 \
	  --bucket-weights 1.1 1.0 1.3 6.2 8.8 --context --fusion film --tta

figures:          ## regenerate docs/figures from the saved runs
	$(PY) scripts/11_figures.py

thesis:           ## rebuild and check the thesis document (needs docs/thesis/content, not in git)
	$(PY) scripts/13_build_thesis.py && $(PY) scripts/14_check_thesis.py

test:
	$(PY) -m pytest -q

smoke:            ## tests that hit EPA and Earth Engine
	$(PY) -m pytest -q -m network tests/test_smoke.py

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'
