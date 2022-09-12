# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.16.0](https://github.com/Substra/substra-tools/releases/tag/0.16.0) - 2022-09-12
### Changed

- Remove documentation as it is not used. It will be replaced later on.
- BREAKING CHANGES: the user must now pass the method name to execute within the dockerfile of both `algo` and
  `metric` under the `--method-name` argument. The method name still needs to be one of the `algo` or `metric`
  allowed method name: train, predict, aggregate, score.

  ```Dockerfile
  ENTRYPOINT ["python3", "metrics.py"]
  ```

  shall be replaced by:

  ```Dockerfile
  ENTRYPOINT ["python3", "metrics.py", "--method-name", "score"]
  ```

- BREAKING CHANGES: rename connect-tools to substra-tools (except the github folder)

## [0.15.0](https://github.com/Substra/substra-tools/releases/tag/0.15.0) - 2022-08-29

### Changed

- BREAKING CHANGES:

  - methods from algo, composite algo, aggregate and metrics now take `inputs` (TypeDict) and `outputs` (TypeDict) as arguments
  - the user must load and save all the inputs and outputs of those methods (except for the datasamples)
  - `load_predictions` and `get_predictions` methods have been removed from the opener
  - `load_trunk_model`, `save_trunk_model`, `load_head_model`, `save_head_model` have been removed from the `tools.CompositeAlgo` class
  - `load_model` and `save_model` have been removed from both `tools.Algo` and `tools.AggregateAlgo` classes

## [0.14.0](https://github.com/Substra/substra-tools/releases/tag/0.14.0) - 2022-08-09

### Changed

- BREAKING CHANGE: drop Python 3.7 support

### Fixed

- fix: metric with type np.float32() is not Json serializable #47

## [0.13.0](https://github.com/Substra/substra-tools/releases/tag/0.13.0) - 2022-05-22

### Changed

- BREAKING CHANGE: change --debug (bool) to --log-level (str)

## [0.12.0](https://github.com/Substra/substra-tools/releases/tag/0.12.0) - 2022-04-29

### Fixed

- nvidia rotating keys

### Changed

- (BREAKING) algos receive arguments are generic inputs/outputs dict

## [0.11.0](https://github.com/Substra/substra-tools/releases/tag/0.11.0) - 2022-04-11

### Fixed

- alias in pyhton 3.7 for python3

### Improved

- ci: build docker images as part of CI checks
- ci: push latest image from main branch
- chore: make Dockerfiles independent from each other
