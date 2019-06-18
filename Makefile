.PHONY: pyclean build test

IMAGE = eu.gcr.io/substra-208412/substratools
TAG = $(shell git log -1 --pretty=format:"%H")

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

build: pyclean
	docker build -t $(IMAGE):$(TAG) .
	docker tag $(IMAGE):$(TAG) $(IMAGE):latest

test: build
	python setup.py test
