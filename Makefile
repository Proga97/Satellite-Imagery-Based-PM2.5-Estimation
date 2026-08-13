PY := /opt/anaconda3/envs/thesis0/bin/python

.PHONY: check labels patches-proto embeddings dataset train test smoke

check:
	$(PY) scripts/00_check_env.py

labels:
	$(PY) scripts/01_build_labels.py

patches-proto:
	$(PY) scripts/02_download_patches.py --limit-stations 10 --limit-weeks 8

embeddings:
	$(PY) scripts/03_extract_embeddings.py

dataset:
	$(PY) scripts/04_assemble_dataset.py

train:
	$(PY) scripts/05_train_eval.py --experiment phase1_image_only

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m pytest -q -m network tests/test_smoke.py
