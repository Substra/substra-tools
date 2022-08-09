# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.14.0](https://github.com/owkin/connect-tools/releases/tag/0.14.0) - 2022-08-09

### Changed

- BREAKING CHANGE: drop Python 3.7 support (#49)

### Fixed

- fix: metric with type np.float32() is not Json serializable #47

## [0.13.0](https://github.com/owkin/connect-tools/releases/tag/0.13.0) - 2022-05-22

### Changed

- BREAKING CHANGE: change --debug (bool) to --log-level (str) (#42)

## [0.12.0](https://github.com/owkin/connect-tools/releases/tag/0.12.0) - 2022-04-29

### Fixed

- nvidia rotating keys (#39)

### Changed

- (BREAKING) algos receive arguments are generic inputs/outputs dict (#27)

## [0.11.0](https://github.com/owkin/connect-tools/releases/tag/0.11.0) - 2022-04-11

### Fixed

- alias in pyhton 3.7 for python3 (#31)

### Improved

- ci: build docker images as part of CI checks (#33)
- ci: push latest image from main branch (#29)
- chore: make Dockerfiles independent from each other (#28)
