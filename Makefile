.PHONY: pyclean build test doc

IMAGE = ghcr.io/substra/substra-tools
DOCS_FILEPATH = docs/api.md

ifeq ($(TAG),)
	TAG := $(shell git describe --always --tags)
endif

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

build: pyclean
	docker build -t $(IMAGE):$(TAG) .

test:
	pytest tests
