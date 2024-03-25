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
docker pull ghcr.io/substra/substra-tools:0.16.0-nvidiacuda11.8.0-base-ubuntu22.04-python3.11
```

## Developers

Clone the repository: <https://github.com/Substra/substra-tools>

### Build the Docker image from source

```sh
docker build -f Dockerfile .
```

### Setup

To setup the project in development mode, run:

```sh
pip install -e ".[dev]"
```

To run all tests, use the following command:

```sh
make test
```

## How to generate the changelog

The changelog is managed with [towncrier](https://towncrier.readthedocs.io/en/stable/index.html).
To add a new entry in the changelog, add a file in the `changes` folder. The file name should have the following structure:
`<unique_id>.<change_type>`.
The `unique_id` is a unique identifier, we currently use the PR number.
The `change_type` can be of the following types: `added`, `changed`, `removed`, `fixed`.

To generate the changelog (for example during a release), use the following command (you must have the dev dependencies installed):

```
towncrier build --version=<x.y.z>
```

You can use the `--draft` option to see what would be generated without actually writing to the changelog (and without removing the fragments).
