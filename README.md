# Substratools

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

## Build substratools image

This is required to launch the substra framework for development. The image is
currently based on Python 3.6.

### Pull from private docker registry

- Install Google Cloud SDK: https://cloud.google.com/sdk/install
- Authenticate with your google account: `gcloud auth login`
- Configure docker to use your google credentials for google based docker registery: `gcloud auth configure-docker`
- Pull image: `docker pull eu.gcr.io/substra-208412/substratools`

### Build from source

```sh
docker build -t eu.gcr.io/substra-208412/substratools .
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
