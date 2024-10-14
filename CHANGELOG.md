# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [1.0.0](https://github.com/Substra/substra-tools/releases/tag/1.0.0) - 2024-10-14

### Removed

- Drop Python 3.9 support. ([#112](https://github.com/Substra/substra-tools/pull/112))


## [0.22.0](https://github.com/Substra/substra-tools/releases/tag/0.22.0) - 2024-09-04

### Changed

- The Opener and Function workspace does not try to create folder for data samples anymore. ([#100](https://github.com/Substra/substra-tools/pull/100))

### Removed

- Remove base Docker image. Substrafl now uses python-slim as base image. ([#101](https://github.com/Substra/substra-tools/pull/101))


## [0.21.4](https://github.com/Substra/substra-tools/releases/tag/0.21.4) - 2024-06-03


No significant changes.


## [0.21.3](https://github.com/Substra/substra-tools/releases/tag/0.21.3) - 2024-03-27


### Changed

- - Drop Python 3.8 support ([#90](https://github.com/Substra/substra-tools/pull/90))
- - Depreciate `setup.py` in favour of `pyproject.toml` ([#92](https://github.com/Substra/substra-tools/pull/92))


## [0.21.2](https://github.com/Substra/substra-tools/releases/tag/0.21.2) - 2024-03-07

### Changed

- Update dependencies

## [0.21.1](https://github.com/Substra/substra-tools/releases/tag/0.21.1) - 2024-02-26

### Changed

- Updated dependencies

## [0.21.0](https://github.com/Substra/substra-tools/releases/tag/0.21.0) - 2023-10-06

### Changed

- Remove `model` and `models` for input and output identifiers in tests. Replace by `shared` instead. ([#84](https://github.com/Substra/substra-tools/pull/84))
- BREAKING: Remove minimal and workflow docker images ([#86](https://github.com/Substra/substra-tools/pull/86))
- Remove python lib from Docker image ([#86](https://github.com/Substra/substra-tools/pull/86))

### Added

- Support on Python 3.11 ([#85](https://github.com/Substra/substra-tools/pull/85))
- Contributing, contributors & code of conduct files (#77)

## [0.20.0](https://github.com/Substra/substra-tools/releases/tag/0.20.0) - 2022-12-19

### Changed

- Add optional argument to `register` decorator to choose a custom function name. Allow to register the same function several time with a different name (#74)
- Rank is now passed in a task properties dictionary from the backend (instead of the rank argument) (#75)

## [0.19.0](https://github.com/Substra/substra-tools/releases/tag/0.19.0) - 2022-11-22

### CHANGED

- BREAKING CHANGE (#65)

  - Register functions to substratools can be done with a decorator.

  ```py
  def my_function1:
    pass

  def my_function2:
    pass

  if __name__ == '__main__':
    tools.execute(my_function1, my_function2)
  ```

  become

  ```py
  @tools.register
  def my_function1:
    pass

  @tools.register
  def my_function2:
    pass

  if __name__ == '__main__':
    tools.execute()
  ```

- BREAKING CHANGE (#63)

  - Rename algo to function.
  - `tools.algo.execute` become `tools.execute`
  - The previous algo class pass to the function `tools.algo.execute` is now several functions pass as arguments to `tools.execute`. The function given by the cli `--function-name` is executed.

  ```py
  if __name__ == '__main__':
    tools.algo.execute(MyAlgo())
  ```

  become

  ```py
  if __name__ == '__main__':
    tools.execute(my_function1, my_function2)
  ```

### Fixed

- Remove depreciated `pytest-runner` from setup.py (#71)
- Replace backslash by slash in TaskResources to fix windows compatibility (#70)
- Update flake8 repository in pre-commit configuration (#69)
- BREAKING CHANGE: Update substratools Docker image (#112)

## [0.18.0](https://github.com/Substra/substra-tools/releases/tag/0.18.0) - 2022-09-26

### Added

- feat: allow CLI parameters to be read from a file

### Changed

- BREAKING CHANGES:

  - the opener only exposes `get_data` and `fake_data` methods.
  - the results of the above method is passed under the `datasamples` keys within the `inputs` dict arg of all
    tools methods (train, predict, aggregate, score).
  - all method (train, predict, aggregate, score) now takes a `task_properties` argument (dict) in addition to
    `inputs` and `outputs`.
  - The `rank` of a task previously passed under the `rank` key within the inputs is now given in the `task_properties`
    dict under the `rank` key.

- BREAKING CHANGE: The metric is now a generic algo, replace

```python
import substratools as tools

class MyMetric(tools.Metrics):
    # ...

if __name__ == '__main__':
    tools.metrics.execute(MyMetric())
```

by

```python
import substratools as tools

class MyMetric(tools.MetricAlgo):
    # ...
if __name__ == '__main__':
    tools.algo.execute(MyMetric())
```

## [0.17.0](https://github.com/Substra/substra-tools/releases/tag/0.17.0) - 2022-09-19

### Changed

- feat: all algo classes rely on a generic algo class

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
