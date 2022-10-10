# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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

## [0.18.0](https://github.com/Substra/substra-tools/releases/tag/0.18.0) - 2022-09-26

### Added

- feat: allow CLI parameters to be read from a file

### CHANGED

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
