IMAGE := pneumonia-xray-classifier
PORT  := 9696

.PHONY: install notebook train serve test predict docker-build docker-run

install:        ## create .venv and install all dependencies from uv.lock
	uv sync

notebook:       ## re-execute the notebook in place
	uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook.ipynb

train:          ## train the final model and export models/pneumonia_classifier.onnx
	uv run python train.py

serve:          ## run the web service locally (no Docker)
	uv run uvicorn predict:app --host 0.0.0.0 --port $(PORT)

test:           ## run API tests
	uv run pytest -q

predict:        ## send sample images to the running service
	uv run python test_predict.py --url http://localhost:$(PORT)

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):9696 $(IMAGE)
