.PHONY: generate train api frontend all clean

generate:
	python -m wafer_sim.generator

train:
	python -m ml.train --data data/wafer_summary.parquet --output-dir ml/artifacts --plots-dir ml/plots

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	npm --prefix frontend run dev

all: generate train

clean:
	rm -rf data ml/artifacts
