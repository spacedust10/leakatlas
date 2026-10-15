# LeakAtlas. `make reproduce` rebuilds every table and figure from data/leakatlas.duckdb (no training, no network).
# `make train` runs the full pipeline from raw data (~2.5 h on an M3 Pro; models are cached and resumable).
PY := PYTHONPATH=src PYTHONWARNINGS=ignore uv run python

.PHONY: train models audit reproduce tables figures serve test

train: models audit reproduce

models:
	$(PY) -m leakatlas.defenses configs/adult.yaml
	$(PY) -m leakatlas.defenses configs/fmnist.yaml

audit:
	$(PY) -m leakatlas.attacks configs/adult.yaml
	$(PY) -m leakatlas.attacks configs/fmnist.yaml

reproduce: tables figures

tables:
	$(PY) -m leakatlas.proxies
	$(PY) -m leakatlas.analysis

figures:
	$(PY) scripts/screenshots.py

serve:
	PYTHONPATH=src uv run panel serve src/leakatlas/viewer.py --show

test:
	$(PY) tests/test_log_grad_norm.py
