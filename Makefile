PY ?= python

.PHONY: data model all ingest test clean-layers

data:
	$(PY) -m src.build

model:
	$(PY) -m src.model_run

all: data model

ingest:
	$(PY) -m src.acquire ingest $(SRC)

test:
	$(PY) -m pytest -q

clean-layers:
	rm -rf data/cleaned data/processed data/model data/_staging
