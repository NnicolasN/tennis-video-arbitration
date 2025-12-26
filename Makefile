.PHONY: install dev test lint format

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

format:
	ruff format .
	black .
