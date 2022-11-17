# Substra-tools

Python package defining base classes for Dataset (data opener script) and wrappers to execute functions submitted on the platform.

This repository also contains a [Dockerfile](https://github.com/Substra/substra-tools/pkgs/container/substra-tools) to execute the user
Python scripts on the Substra platform. This is currently needed to easily
have substratools package available inside the Docker image without using a
pypi server.

## Getting started

To install the substratools Python package, run the following command:

```sh
pip install substratools
```

## Pull the substra-tools Docker image

```sh
docker pull ghcr.io/substra/substra-tools:0.16.0-nvidiacuda11.8.0-base-ubuntu22.04-python3.10-workflows
```

## Developers

Clone the repository: <https://github.com/Substra/substra-tools>

### Build the Docker image from source

```sh
docker build -f Dockerfile .
```

or for the minimal image (based on alpine):

```sh
docker build -f Dockerfile.minimal .
```

or for the workflows image (contains additional data science dependencies):

```sh
docker build -f Dockerfile.workflows .
```

### Setup

To setup the project in development mode, run:

```sh
pip install -e ".[test]"
```

To run all tests, use the following command:

```sh
python setup.py test
```
