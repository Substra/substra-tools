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
	docker build -f Dockerfile.minimal -t $(IMAGE):test_gt_arm64-minimal .
	docker push $(IMAGE):test_gt_arm64-minimal
	docker build -f Dockerfile -t $(IMAGE):test_gt_arm64 .
	docker push $(IMAGE):test_gt_arm64
	docker build -f Dockerfile.workflows -t $(IMAGE):test_gt_arm64-workflows .
	docker push $(IMAGE):test_gt_arm64-workflows


test:
	pytest tests
