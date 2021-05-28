.PHONY: pyclean build test doc

IMAGE = gcr.io/connect-314908/connect-tools
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
	python setup.py test

doc:
	python -m pydoc substratools.Algo substratools.CompositeAlgo substratools.AggregateAlgo substratools.Metrics substratools.Opener > $(DOCS_FILEPATH)
	# Replace '| ' by '' to make the doc readable. Portable solution that works on Linux and Mac OS
	sed -i.bak 's/| //g' '$(DOCS_FILEPATH)' && rm $(DOCS_FILEPATH).bak
