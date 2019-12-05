# Substra-tools

Python package defining base classes for assets submitted on the platform:
- Objective: metrics script
- Algo: algo script
- Dataset: data opener script

This repository also contains a [Dockerfile](Dockerfile) to execute the user
python scripts on the Substra platform. This is currently needed to easily
have substratools package available inside the Docker image without using a
pypi server.

## Getting started

To install the substratools python package, run the following command:

```sh
pip install .
```

## Documentation

- [API](docs/api.md)

## Build substra-tools image

This is required to launch the substra framework for development. The image is
currently based on Python 3.6.

### Pull from public docker registry

```sh
docker pull substrafoundation/substra-tools
```

### Build from source

```sh
docker build -t substrafoundation/substra-tools .
```

## Contributing
### Setup

To setup the project in development mode, run:

```sh
pip install -e .[test]
```

To run all tests, use the following command:

```sh
python setup.py test
```

### Documentation

Use the following command to generate the python sdk documentation:

```sh
pydocmd simple substratools.Algo+ substratools.Metrics+ substratools.Opener+> docs/api.md
```

Documentation will be available in *docs/* directory.
