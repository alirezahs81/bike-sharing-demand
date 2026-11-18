# Makefile for Bike Sharing Demand Prediction
# Usage: make <target>

.PHONY: help install install-dev setup download-data preprocess train test lint format type-check api docker-build docker-run clean

help:
	@echo "Bike Sharing Demand - Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  setup          Full setup (install + download + preprocess + train)"
	@echo ""
	@echo "Data & Training:"
	@echo "  download-data  Download dataset from UCI"
	@echo "  preprocess     Run data preprocessing"
	@echo "  train          Train all models"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test           Run unit tests with coverage"
	@echo "  lint           Run ruff linter"
	@echo "  format         Format code with ruff"
	@echo "  type-check     Run mypy type checking"
	@echo ""
	@echo "API & Docker:"
	@echo "  api            Run FastAPI server locally"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          Remove cache, build artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

setup: install download-data preprocess train
	@echo "Setup complete! Run 'make api' to start the server."

download-data:
	python scripts/download_data.py

preprocess:
	python src/preprocess.py

train:
	python src/train.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

type-check:
	mypy src/

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t bike-sharing-demand .

docker-run:
	docker run -p 8000:8000 bike-sharing-demand

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf src/__pycache__ tests/__pycache__ scripts/__pycache__
	rm -rf *.egg-info dist build
