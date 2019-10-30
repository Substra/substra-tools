.PHONY: pyclean build test doc

IMAGE = substrafoundation/substra-tools

ifeq ($(TAG),)
	TAG := $(shell git describe --always --tags)
endif

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

build: pyclean
	docker build -t $(IMAGE):$(TAG) .

test:
	python setup.py test

doc:
	pydocmd simple substratools.Algo+ substratools.CompositeAlgo+ substratools.Metrics+ substratools.Opener+> docs/api.md
