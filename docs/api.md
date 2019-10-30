# Algo
```python
Algo(self, /, *args, **kwargs)
```
Abstract base class for defining algo to run on the platform.

To define a new algo script, subclass this class and implement the
following abstract methods:

- `Algo.train()`
- `Algo.predict()`
- `Algo.load_model()`
- `Algo.save_model()`

To add an algo to the Substra Platform, the line
`tools.algo.execute(<AlgoClass>())` must be added to the main of the algo
python script. It defines the algo command line interface and thus enables
the Substra Platform to execute it.

__Example__


```python
import json
import substratools as tools


class DummyAlgo(tools.Algo):
    def train(self, X, y, models, rank):
        predictions = 0
        new_model = None
        return predictions, new_model

    def predict(self, X, model):
        predictions = 0
        return predictions

    def load_model(self, path):
        return json.load(path)

    def save_model(self, model, path):
        json.dump(model, path)


if __name__ == '__main__':
    tools.algo.execute(DummyAlgo())
```

__How to test locally an algo script__


The algo script can be directly tested through it's command line interface.
For instance to train an algo using fake data, run the following command:

```sh
python <script_path> train --fake-data --debug
```

To see all the available options for the train and predict commands, run:

```sh
python <script_path> train --help
python <script_path> predict --help
```


## train
```python
Algo.train(self, X, y, models, rank)
```
Train model and produce new model from train data.

This task corresponds to the creation of a traintuple on the Substra
Platform.

__Arguments__


- __X__: training data samples loaded with `Opener.get_X()`.
- __y__: training data samples labels loaded with `Opener.get_y()`.
- __models__: list of models loaded with `Algo.load_model()`.
- __rank__: rank of the training task.

__Returns__


`tuple`: (predictions, model).

## predict
```python
Algo.predict(self, X, model)
```
Get predictions from test data.

This task corresponds to the creation of a testtuple on the Substra
Platform.

__Arguments__


- __X__: testing data samples loaded with `Opener.get_X()`.
- __model__: input model load with `Algo.load_model()` used for predictions.

__Returns__


`predictions`: predictions object.

## load_model
```python
Algo.load_model(self, path)
```
Deserialize model from file.

This method will be executed before the call to the methods
`Algo.train()` and `Algo.predict()` to deserialize the model objects.

__Arguments__


- __path__: path of the model to load.

__Returns__


`model`: the deserialized model object.

## save_model
```python
Algo.save_model(self, model, path)
```
Serialize model in file.

This method will be executed after the call to the methods
`Algo.train()` and `Algo.predict()` to save the model objects.

__Arguments__


- __path__: path of file to write.
- __model__: the model to serialize.

# CompositeAlgo
```python
CompositeAlgo(self, /, *args, **kwargs)
```
Abstract base class for defining a composite algo to run on the platform.

To define a new composite algo script, subclass this class and implement the
following abstract methods:

- `CompositeAlgo.train()`
- `CompositeAlgo.predict()`
- `CompositeAlgo.load_model()`
- `CompositeAlgo.save_model()`

To add a composite algo to the Substra Platform, the line
`tools.algo.execute(<AlgoClass>())` must be added to the main of the algo
python script. It defines the composite algo command line interface and thus enables
the Substra Platform to execute it.

__Example__


```python
import json
import substratools as tools


class DummyCompositeAlgo(tools.CompositeAlgo):
    def train(self, X, y, head_model, trunk_model, rank):
        predictions = 0
        new_head_model = None
        new_trunk_model = None
        return predictions, new_head_model, new_trunk_model

    def predict(self, X, head_model, trunk_model):
        predictions = 0
        return predictions

    def load_model(self, path):
        return json.load(path)

    def save_model(self, model, path):
        json.dump(model, path)


if __name__ == '__main__':
    tools.algo.execute(DummyCompositeAlgo())
```

## train
```python
CompositeAlgo.train(self, X, y, head_model, trunk_model, rank)
```
Train model and produce new composite models from train data.

This task corresponds to the creation of a composite traintuple on the Substra
Platform.

__Arguments__


- __X__: training data samples loaded with `Opener.get_X()`.
- __y__: training data samples labels loaded with `Opener.get_y()`.
- __head_model__: head model loaded with `Algo.load_model()` (may be None).
- __trunk_model__: trunk model loaded with `Algo.load_model()` (may be None).
- __rank__: rank of the training task.

__Returns__


`tuple`: (predictions, head_model, trunk_model).

## predict
```python
CompositeAlgo.predict(self, X, head_model, trunk_model)
```
Get predictions from test data.

This task corresponds to the creation of a composite testtuple on the Substra
Platform.

__Arguments__


- __X__: testing data samples loaded with `Opener.get_X()`.
- __head_model__: head model loaded with `Algo.load_model()`.
- __trunk_model__: trunk model loaded with `Algo.load_model()`.

__Returns__


`predictions`: predictions object.

# Metrics
```python
Metrics(self, /, *args, **kwargs)
```
Abstract base class for defining the objective metrics.

To define a new metrics, subclass this class and implement the
unique following abstract method `Metrics.score()`.

To add an objective to the Substra Platform, the line
`tools.algo.execute(<MetricsClass>())` must be added to the main of the
metrics python script. It defines the metrics command line interface and
thus enables the Substra Platform to execute it.

__Example__


```python
from sklearn.metrics import accuracy_score
import substratools as tools


class AccuracyMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

if __name__ == '__main__':
     tools.metrics.execute(AccuracyMetrics())
```

__How to test locally a metrics script__


The metrics script can be directly tested through it's command line
interface.  For instance to get the metrics from fake data, run the
following command:

```sh
python <script_path> --fake-data --debug
```

To see all the available options for metrics commands, run:

```sh
python <script_path> --help
```


## score
```python
Metrics.score(self, y_true, y_pred)
```
Compute model perf from actual and predicted values.

__Arguments__


- __y_true__: actual values.
- __y_pred__: predicted values.

__Returns__


`perf (float)`: performance of the model.

# Opener
```python
Opener(self, /, *args, **kwargs)
```
Dataset opener abstract base class.

To define a new opener script, subclass this class and implement the
following abstract methods:

- `Opener.get_X()`
- `Opener.get_y()`
- `Opener.fake_X()`
- `Opener.fake_y()`
- `Opener.get_predictions()`
- `Opener.save_predictions()`

__Example__


```python
import os
import pandas as pd
import string
import numpy as np

import substratools as tools

class DummyOpener(tools.Opener):
    def get_X(self, folders):
        return [
            pd.read_csv(os.path.join(folder, 'train.csv'))
            for folder in folders
        ]

    def get_y(self, folders):
        return [
            pd.read_csv(os.path.join(folder, 'y.csv'))
            for folder in folders
        ]

    def fake_X(self):
        return []  # compute random fake data

    def fake_y(self):
        return []  # compute random fake data

    def save_predictions(self, y_pred, path):
        with open(path, 'w') as fp:
            y_pred.to_csv(fp, index=False)

    def get_predictions(self, path):
        return pd.read_csv(path)
```

## get_X
```python
Opener.get_X(self, folders)
```
Load feature data from data sample folders.

__Arguments__


- __folders__: list of folders. Each folder represents a data sample.

__Returns__


`data`: data object.

## get_y
```python
Opener.get_y(self, folders)
```
Load labels from data sample folders.

__Arguments__


- __folders__: list of folders. Each folder represents a data sample.

__Returns__


`data`: data labels object.

## fake_X
```python
Opener.fake_X(self)
```
Generate a fake matrix of features for offline testing.

__Returns__


`data`: data labels object.

## fake_y
```python
Opener.fake_y(self)
```
Generate a fake target variable vector for offline testing.

__Returns__


`data`: data labels object.

## get_predictions
```python
Opener.get_predictions(self, path)
```
Read file and return predictions vector.

__Arguments__


- __path__: string file path.

__Returns__


`predictions`: predictions vector.

## save_predictions
```python
Opener.save_predictions(self, y_pred, path)
```
Write predictions vector to file.

__Arguments__


- __y_pred__: predictions vector.
- __path__: string file path.

