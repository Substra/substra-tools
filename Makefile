.PHONY: test doc

DOCS_FILEPATH = docs/api.md

ifeq ($(TAG),)
	TAG := $(shell git describe --always --tags)
endif

test:
	pytest tests
