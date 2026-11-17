.PHONY: all test lint type

all: lint type test

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run mypy --strict src tests
