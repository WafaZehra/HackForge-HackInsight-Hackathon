PY ?= python

.PHONY: data ingest test clean-layers

data:
	$(PY) -m src.build

ingest:
	$(PY) -m src.acquire ingest $(SRC)

test:
	$(PY) -m pytest -q

clean-layers:
	rm -rf data/cleaned data/processed data/_staging
