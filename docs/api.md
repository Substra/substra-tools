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

This class has an `use_models_generator` class property:
- if True, models will be passed to the `train` method as a generator
- (default) if False, models will be passed to the `train` method as a list

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
        new_model = None
        return new_model

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


__Using the command line__


The algo script can be directly tested through it's command line interface.
For instance to train an algo using fake data, run the following command:

```sh
python <script_path> train --fake-data --n-fake-samples 20 --debug
```

To see all the available options for the train and predict commands, run:

```sh
python <script_path> train --help
python <script_path> predict --help
```

__Using a python script__


An algo can be imported and used in python scripts as would any other class.

For example, assuming that you have two local files named `opener.py` and
`algo.py` (the latter containing an `Algo` class named `MyAlgo`):

```python
import algo
import opener

o = opener.Opener()
X = o.get_X(["dataset/train/train1"])
y = o.get_y(["dataset/train/train1"])

a = algo.MyAlgo()
model = a.train(X, y, None, None, 0)
y_pred = a.predict(X, model)
```


## use_models_generator
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.
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
- __models__: list or generator of models loaded with `Algo.load_model()`.
- __rank__: rank of the training task.

__Returns__


`model`: model object.

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
- `CompositeAlgo.load_head_model()`
- `CompositeAlgo.save_head_model()`
- `CompositeAlgo.load_trunk_model()`
- `CompositeAlgo.save_trunk_model()`

To add a composite algo to the Substra Platform, the line
`tools.algo.execute(<CompositeAlgoClass>())` must be added to the main of the algo
python script. It defines the composite algo command line interface and thus enables
the Substra Platform to execute it.

__Example__


```python
import json
import substratools as tools


class DummyCompositeAlgo(tools.CompositeAlgo):
    def train(self, X, y, head_model, trunk_model, rank):
        new_head_model = None
        new_trunk_model = None
        return new_head_model, new_trunk_model

    def predict(self, X, head_model, trunk_model):
        predictions = 0
        return predictions

    def load_head_model(self, path):
        return json.load(path)

    def save_head_model(self, model, path):
        json.dump(model, path)

    def load_trunk_model(self, path):
        return json.load(path)

    def save_trunk_model(self, model, path):
        json.dump(model, path)


if __name__ == '__main__':
    tools.algo.execute(DummyCompositeAlgo())
```

__How to test locally a composite algo script__


__Using the command line__


The composite algo script can be directly tested through it's command line interface.
For instance to train an algo using fake data, run the following command:

```sh
python <script_path> train --fake-data --n-fake-samples 20 --debug
```

To see all the available options for the train and predict commands, run:

```sh
python <script_path> train --help
python <script_path> predict --help
```

__Using a python script__


A composite algo can be imported and used in python scripts as would any other class.

For example, assuming that you have two local files named `opener.py` and
`composite_algo.py` (the latter containing a `CompositeAlgo` class named
`MyCompositeAlgo`):

```python
import composite_algo
import opener

o = opener.Opener()
X = o.get_X(["dataset/train/train1"])
y = o.get_y(["dataset/train/train1"])

a = composite_algo.MyCompositeAlgo()
head_model, trunk_model = a.train(X, y, None, None, 0)
y_pred = a.predict(X, head_model, trunk_model)
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
- __head_model__: head model loaded with `CompositeAlgo.load_head_model()` (may be None).
- __trunk_model__: trunk model loaded with `CompositeAlgo.load_trunk_model()` (may be None).
- __rank__: rank of the training task.

__Returns__


`tuple`: (head_model, trunk_model).

## predict
```python
CompositeAlgo.predict(self, X, head_model, trunk_model)
```
Get predictions from test data.

This task corresponds to the creation of a composite testtuple on the Substra
Platform.

__Arguments__


- __X__: testing data samples loaded with `Opener.get_X()`.
- __head_model__: head model loaded with `CompositeAlgo.load_head_model()`.
- __trunk_model__: trunk model loaded with `CompositeAlgo.load_trunk_model()`.

__Returns__


`predictions`: predictions object.

## load_head_model
```python
CompositeAlgo.load_head_model(self, path)
```
Deserialize head model from file.

This method will be executed before the call to the methods
`Algo.train()` and `Algo.predict()` to deserialize the model objects.

__Arguments__


- __path__: path of the model to load.

__Returns__


`model`: the deserialized model object.

## save_head_model
```python
CompositeAlgo.save_head_model(self, model, path)
```
Serialize head model in file.

This method will be executed after the call to the methods
`Algo.train()` and `Algo.predict()` to save the model objects.

__Arguments__


- __path__: path of file to write.
- __model__: the model to serialize.

## load_trunk_model
```python
CompositeAlgo.load_trunk_model(self, path)
```
Deserialize trunk model from file.

This method will be executed before the call to the methods
`Algo.train()` and `Algo.predict()` to deserialize the model objects.

__Arguments__


- __path__: path of the model to load.

__Returns__


`model`: the deserialized model object.

## save_trunk_model
```python
CompositeAlgo.save_trunk_model(self, model, path)
```
Serialize trunk model in file.

This method will be executed after the call to the methods
`Algo.train()` and `Algo.predict()` to save the model objects.

__Arguments__


- __path__: path of file to write.
- __model__: the model to serialize.

# AggregateAlgo
```python
AggregateAlgo(self, /, *args, **kwargs)
```
Abstract base class for defining an aggregate algo to run on the platform.

To define a new aggregate algo script, subclass this class and implement the
following abstract methods:

- `AggregateAlgo.aggregate()`
- `AggregateAlgo.load_model()`
- `AggregateAlgo.save_model()`

This class has an `use_models_generator` class property:
- if True, models will be passed to the `aggregate` method as a generator
- (default) if False, models will be passed to the `aggregate` method as a list

To add a aggregate algo to the Substra Platform, the line
`tools.algo.execute(<AggregateAlgoClass>())` must be added to the main of the algo
python script. It defines the aggregate algo command line interface and thus enables
the Substra Platform to execute it.

__Example__


```python
import json
import substratools as tools


class DummyAggregateAlgo(tools.AggregateAlgo):
    def aggregate(self, models, rank):
        new_model = None
        return new_model

    def load_model(self, path):
        return json.load(path)

    def save_model(self, model, path):
        json.dump(model, path)


if __name__ == '__main__':
    tools.algo.execute(DummyAggregateAlgo())
```

__How to test locally an aggregate algo script__


__Using the command line__


The aggregate algo script can be directly tested through it's command line interface.
For instance to train an algo using fake data, run the following command:

```sh
python <script_path> aggregate --models_path <models_path> --models <model_name> --model <model_name> --debug
```

To see all the available options for the aggregate command, run:

```sh
python <script_path> aggregate --help
```

__Using a python script__


An aggregate algo can be imported and used in python scripts as would any other class.

For example, assuming that you have a local file named `aggregate_algo.py` containing
containing an `AggregateAlgo` class named `MyAggregateAlgo`:

```python
from aggregate_algo import MyAggregateAlgo

a = MyAggregateAlgo()

model_1 = a.load_model('./sandbox/models/model_1')
model_2 = a.load_model('./sandbox/models/model_2')

aggregated_model = a.aggregate([model_1, model_2], 0)
```

## use_models_generator
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.
## aggregate
```python
AggregateAlgo.aggregate(self, models, rank)
```
Aggregate models and produce a new model.

This task corresponds to the creation of an aggregate tuple on the Substra
Platform.

__Arguments__


- __models__: list of models loaded with `AggregateAlgo.load_model()`.
- __rank__: rank of the aggregate task.

__Returns__


`model`: aggregated model.

## load_model
```python
AggregateAlgo.load_model(self, path)
```
Deserialize model from file.

This method will be executed before the call to the method `Algo.aggregate()`
to deserialize the model objects.

__Arguments__


- __path__: path of the model to load.

__Returns__


`model`: the deserialized model object.

## save_model
```python
AggregateAlgo.save_model(self, model, path)
```
Serialize model in file.

This method will be executed after the call to the method `Algo.aggregate()`
to save the model objects.

__Arguments__


- __path__: path of file to write.
- __model__: the model to serialize.

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


__Using the command line__


The metrics script can be directly tested through it's command line
interface.  For instance to get the metrics from fake data, run the
following command:

```sh
python <script_path> --fake-data --n-fake-samples 20 --debug
```

To see all the available options for metrics commands, run:

```sh
python <script_path> --help
```

__Using a python script__


A metrics class can be imported and used in python scripts as would any other class.

For example, assuming that you have files named `opener.py` and `metrics.py` that contains
an `Opener` named  `MyOpener` and a `Metrics` called `MyMetrics`:

```python
import os
import opener
import metrics

o = MyOpener()
m = MyMetrics()


data_sample_folders = os.listdir('./sandbox/data_samples/')
predictions_path = './sandbox/predictions'

y_true = o.get_y(data_sample_folders)
y_pred = o.get_predictions(predictions_path)

score = m.score(y_true, y_pred)
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

    def fake_X(self, n_samples=None):
        return []  # compute random fake data

    def fake_y(self, n_samples=None):
        return []  # compute random fake data

    def save_predictions(self, y_pred, path):
        with open(path, 'w') as fp:
            y_pred.to_csv(fp, index=False)

    def get_predictions(self, path):
        return pd.read_csv(path)
```

__How to test locally an opener script__


An opener can be imported and used in python scripts as would any other class.

For example, assuming that you have a local file named `opener.py` that contains
an `Opener` named  `MyOpener`:

```python
import os
from opener import MyOpener

folders = os.listdir('./sandbox/data_samples/')

o = MyOpener()
X = o.get_X(folders)
y = o.get_y(folders)
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
Opener.fake_X(self, n_samples=None)
```
Generate a fake matrix of features for offline testing.

__Arguments__


- __n_samples__: number of samples to return

__Returns__


`data`: data labels object.

## fake_y
```python
Opener.fake_y(self, n_samples=None)
```
Generate a fake target variable vector for offline testing.

__Arguments__


- __n_samples__: number of samples to return

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

