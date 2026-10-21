PY ?= python

.PHONY: data ingest test clean-layers

# Rebuild raw-check -> cleaned -> processed from scratch. First run: put Data.zip in data/incoming/.
data:
	$(PY) -m src.build

# One-off, optional: ingest explicitly from a zip or folder.  make ingest SRC=path/to/Data.zip
ingest:
	$(PY) -m src.acquire ingest $(SRC)

test:
	$(PY) -m pytest -q

# Delete derived layers only. Raw files and manifest.csv are never touched.
clean-layers:
	rm -rf data/cleaned data/processed data/_staging
