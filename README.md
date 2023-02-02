# Substra-tools

<div align="left">
<a href="https://join.slack.com/t/substra-workspace/shared_invite/zt-1fqnk0nw6-xoPwuLJ8dAPXThfyldX8yA"><img src="https://img.shields.io/badge/chat-on%20slack-blue?logo=slack" /></a> <a href="https://docs.substra.org/en/stable/documentation/substra_tools.html"><img src="https://img.shields.io/badge/read-docs-purple?logo=mdbook" /></a>
<br /><br /></div>

<div align="center">
<picture>
  <object-position: center>
  <source media="(prefers-color-scheme: dark)" srcset="Substra-logo-white.svg">
  <source media="(prefers-color-scheme: light)" srcset="Substra-logo-colour.svg">
  <img alt="Substra" src="Substra-logo-colour.svg" width="500">
</picture>
</div>
<br>
<br>

Substra is an open source federated learning (FL) software. This specific repository, substra-tools, is a Python package defining base classes for Dataset (data opener script) and wrappers to execute functions submitted on the platform.


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
