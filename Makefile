.PHONY: pyclean build test

TAG = $(shell git log -1 --pretty=format:"%H")

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

build: pyclean
	docker build -t eu.gcr.io/substra-208412/substratools:$(TAG) .

test: build
	python setup.py test
