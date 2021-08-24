Help on class Algo in substratools:

substratools.Algo = class Algo(abc.ABC)
  Abstract base class for defining algo to run on the platform.
  
  To define a new algo script, subclass this class and implement the
  following abstract methods:
  
  - #Algo.train()
  - #Algo.predict()
  - #Algo.load_model()
  - #Algo.save_model()
  
  This class has an `use_models_generator` class property:
  - if True, models will be passed to the `train` method as a generator
  - (default) if False, models will be passed to the `train` method as a list
  
  The class has a `chainkeys_path` class property: it contains the path to the chainkeys folder.
  If the chainkey support is on, this folder contains the chainkeys.
  
  The class has a `compute_plan_path` class property: it contains the path to the compute plan local folder.
  If the algo is executed as part of a compute plan, this folder contains the shared data between the tasks of
  the compute plan.
  
  To add an algo to the Substra Platform, the line
  `tools.algo.execute(<AlgoClass>())` must be added to the main of the algo
  python script. It defines the algo command line interface and thus enables
  the Substra Platform to execute it.
  
  # Example
  
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
  
  # How to test locally an algo script
  
  # Using the command line
  
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
  
  # Using a python script
  
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
  
  Method resolution order:
      Algo
      abc.ABC
      builtins.object
  
  Methods defined here:
  
  load_model(self, path)
      Deserialize model from file.
      
      This method will be executed before the call to the methods
      `Algo.train()` and `Algo.predict()` to deserialize the model objects.
      
      # Arguments
      
      path: path of the model to load.
      
      # Returns
      
      model: the deserialized model object.
  
  predict(self, X, model)
      Get predictions from test data.
      
      This task corresponds to the creation of a testtuple on the Substra
      Platform.
      
      # Arguments
      
      X: testing data samples loaded with `Opener.get_X()`.
      model: input model load with `Algo.load_model()` used for predictions.
      
      # Returns
      
      predictions: predictions object.
  
  save_model(self, model, path)
      Serialize model in file.
      
      This method will be executed after the call to the methods
      `Algo.train()` and `Algo.predict()` to save the model objects.
      
      # Arguments
      
      path: path of file to write.
      model: the model to serialize.
  
  train(self, X, y, models, rank)
      Train model and produce new model from train data.
      
      This task corresponds to the creation of a traintuple on the Substra
      Platform.
      
      # Arguments
      
      X: training data samples loaded with `Opener.get_X()`.
      y: training data samples labels loaded with `Opener.get_y()`.
      models: list or generator of models loaded with `Algo.load_model()`.
      rank: rank of the training task.
      
      # Returns
      
      model: model object.
  
  ----------------------------------------------------------------------
  Data descriptors defined here:
  
  __dict__
      dictionary for instance variables (if defined)
  
  __weakref__
      list of weak references to the object (if defined)
  
  ----------------------------------------------------------------------
  Data and other attributes defined here:
  
  __abstractmethods__ = frozenset({'load_model', 'predict', 'save_model'...
  
  chainkeys_path = None
  
  compute_plan_path = None
  
  use_models_generator = False

Help on class CompositeAlgo in substratools:

substratools.CompositeAlgo = class CompositeAlgo(abc.ABC)
  Abstract base class for defining a composite algo to run on the platform.
  
  To define a new composite algo script, subclass this class and implement the
  following abstract methods:
  
  - #CompositeAlgo.train()
  - #CompositeAlgo.predict()
  - #CompositeAlgo.load_head_model()
  - #CompositeAlgo.save_head_model()
  - #CompositeAlgo.load_trunk_model()
  - #CompositeAlgo.save_trunk_model()
  
  To add a composite algo to the Substra Platform, the line
  `tools.algo.execute(<CompositeAlgoClass>())` must be added to the main of the algo
  python script. It defines the composite algo command line interface and thus enables
  the Substra Platform to execute it.
  
  The class has a `chainkeys_path` class property: it contains the path to the chainkeys folder.
  If the chainkey support is on, this folder contains the chainkeys.
  
  The class has a `compute_plan_path` class property: it contains the path to the compute plan local folder.
  If the algo is executed as part of a compute plan, this folder contains the shared data between the tasks of
  the compute plan.
  
  # Example
  
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
  
  # How to test locally a composite algo script
  
  # Using the command line
  
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
  
  # Using a python script
  
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
  
  Method resolution order:
      CompositeAlgo
      abc.ABC
      builtins.object
  
  Methods defined here:
  
  load_head_model(self, path)
      Deserialize head model from file.
      
      This method will be executed before the call to the methods
      `Algo.train()` and `Algo.predict()` to deserialize the model objects.
      
      # Arguments
      
      path: path of the model to load.
      
      # Returns
      
      model: the deserialized model object.
  
  load_trunk_model(self, path)
      Deserialize trunk model from file.
      
      This method will be executed before the call to the methods
      `Algo.train()` and `Algo.predict()` to deserialize the model objects.
      
      # Arguments
      
      path: path of the model to load.
      
      # Returns
      
      model: the deserialized model object.
  
  predict(self, X, head_model, trunk_model)
      Get predictions from test data.
      
      This task corresponds to the creation of a composite testtuple on the Substra
      Platform.
      
      # Arguments
      
      X: testing data samples loaded with `Opener.get_X()`.
      head_model: head model loaded with `CompositeAlgo.load_head_model()`.
      trunk_model: trunk model loaded with `CompositeAlgo.load_trunk_model()`.
      
      # Returns
      
      predictions: predictions object.
  
  save_head_model(self, model, path)
      Serialize head model in file.
      
      This method will be executed after the call to the methods
      `Algo.train()` and `Algo.predict()` to save the model objects.
      
      # Arguments
      
      path: path of file to write.
      model: the model to serialize.
  
  save_trunk_model(self, model, path)
      Serialize trunk model in file.
      
      This method will be executed after the call to the methods
      `Algo.train()` and `Algo.predict()` to save the model objects.
      
      # Arguments
      
      path: path of file to write.
      model: the model to serialize.
  
  train(self, X, y, head_model, trunk_model, rank)
      Train model and produce new composite models from train data.
      
      This task corresponds to the creation of a composite traintuple on the Substra
      Platform.
      
      # Arguments
      
      X: training data samples loaded with `Opener.get_X()`.
      y: training data samples labels loaded with `Opener.get_y()`.
      head_model: head model loaded with `CompositeAlgo.load_head_model()` (may be None).
      trunk_model: trunk model loaded with `CompositeAlgo.load_trunk_model()` (may be None).
      rank: rank of the training task.
      
      # Returns
      
      tuple: (head_model, trunk_model).
  
  ----------------------------------------------------------------------
  Data descriptors defined here:
  
  __dict__
      dictionary for instance variables (if defined)
  
  __weakref__
      list of weak references to the object (if defined)
  
  ----------------------------------------------------------------------
  Data and other attributes defined here:
  
  __abstractmethods__ = frozenset({'load_head_model', 'load_trunk_model'...
  
  chainkeys_path = None
  
  compute_plan_path = None

Help on class AggregateAlgo in substratools:

substratools.AggregateAlgo = class AggregateAlgo(abc.ABC)
  Abstract base class for defining an aggregate algo to run on the platform.
  
  To define a new aggregate algo script, subclass this class and implement the
  following abstract methods:
  
  - #AggregateAlgo.aggregate()
  - #AggregateAlgo.predict()
  - #AggregateAlgo.load_model()
  - #AggregateAlgo.save_model()
  
  This class has an `use_models_generator` class property:
  - if True, models will be passed to the `aggregate` method as a generator
  - (default) if False, models will be passed to the `aggregate` method as a list
  
  The class has a `chainkeys_path` class property: it contains the path to the chainkeys folder.
  If the chainkey support is on, this folder contains the chainkeys.
  
  The class has a `compute_plan_path` class property: it contains the path to the compute plan local folder.
  If the algo is executed as part of a compute plan, this folder contains the shared data between the tasks of
  the compute plan.
  
  To add a aggregate algo to the Substra Platform, the line
  `tools.algo.execute(<AggregateAlgoClass>())` must be added to the main of the algo
  python script. It defines the aggregate algo command line interface and thus enables
  the Substra Platform to execute it.
  
  # Example
  
  ```python
  import json
  import substratools as tools
  
  
  class DummyAggregateAlgo(tools.AggregateAlgo):
      def aggregate(self, models, rank):
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
      tools.algo.execute(DummyAggregateAlgo())
  ```
  
  # How to test locally an aggregate algo script
  
  # Using the command line
  
  The aggregate algo script can be directly tested through it's command line interface.
  For instance to train an algo using fake data, run the following command:
  
  ```sh
  python <script_path> aggregate --models_path <models_path> --models <model_name> --model <model_name> --debug
  ```
  
  To see all the available options for the aggregate and predict commands, run:
  
  ```sh
  python <script_path> aggregate --help
  python <script_path> predict --help
  ```
  
  # Using a python script
  
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
  
  Method resolution order:
      AggregateAlgo
      abc.ABC
      builtins.object
  
  Methods defined here:
  
  aggregate(self, models, rank)
      Aggregate models and produce a new model.
      
      This task corresponds to the creation of an aggregate tuple on the Substra
      Platform.
      
      # Arguments
      
      models: list of models loaded with `AggregateAlgo.load_model()`.
      rank: rank of the aggregate task.
      
      # Returns
      
      model: aggregated model.
  
  load_model(self, path)
      Deserialize model from file.
      
      This method will be executed before the call to the method `Algo.aggregate()`
      to deserialize the model objects.
      
      # Arguments
      
      path: path of the model to load.
      
      # Returns
      
      model: the deserialized model object.
  
  predict(self, X, model)
      Get predictions from test data.
      
      This task corresponds to the creation of a testtuple on the Substra
      Platform.
      
      # Arguments
      
      X: testing data samples loaded with `Opener.get_X()`.
      model: input model load with `AggregateAlgo.load_model()` used for predictions.
      
      # Returns
      
      predictions: predictions object.
  
  save_model(self, model, path)
      Serialize model in file.
      
      This method will be executed after the call to the method `Algo.aggregate()`
      to save the model objects.
      
      # Arguments
      
      path: path of file to write.
      model: the model to serialize.
  
  ----------------------------------------------------------------------
  Data descriptors defined here:
  
  __dict__
      dictionary for instance variables (if defined)
  
  __weakref__
      list of weak references to the object (if defined)
  
  ----------------------------------------------------------------------
  Data and other attributes defined here:
  
  __abstractmethods__ = frozenset({'aggregate', 'load_model', 'predict',...
  
  chainkeys_path = None
  
  compute_plan_path = None
  
  use_models_generator = False

Help on class Metrics in substratools:

substratools.Metrics = class Metrics(abc.ABC)
  Abstract base class for defining the objective metrics.
  
  To define a new metrics, subclass this class and implement the
  unique following abstract method #Metrics.score().
  
  To add an objective to the Substra Platform, the line
  `tools.algo.execute(<MetricsClass>())` must be added to the main of the
  metrics python script. It defines the metrics command line interface and
  thus enables the Substra Platform to execute it.
  
  # Example
  
  ```python
  from sklearn.metrics import accuracy_score
  import substratools as tools
  
  
  class AccuracyMetrics(tools.Metrics):
      def score(self, y_true, y_pred):
          return accuracy_score(y_true, y_pred)
  
  if __name__ == '__main__':
       tools.metrics.execute(AccuracyMetrics())
  ```
  
  # How to test locally a metrics script
  
  # Using the command line
  
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
  
  # Using a python script
  
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
  
  Method resolution order:
      Metrics
      abc.ABC
      builtins.object
  
  Methods defined here:
  
  score(self, y_true, y_pred)
      Compute model perf from actual and predicted values.
      
      # Arguments
      
      y_true: actual values.
      y_pred: predicted values.
      
      # Returns
      
      perf (float): performance of the model.
  
  ----------------------------------------------------------------------
  Data descriptors defined here:
  
  __dict__
      dictionary for instance variables (if defined)
  
  __weakref__
      list of weak references to the object (if defined)
  
  ----------------------------------------------------------------------
  Data and other attributes defined here:
  
  __abstractmethods__ = frozenset({'score'})

Help on class Opener in substratools:

substratools.Opener = class Opener(abc.ABC)
  Dataset opener abstract base class.
  
  To define a new opener script, subclass this class and implement the
  following abstract methods:
  
  - #Opener.get_X()
  - #Opener.get_y()
  - #Opener.fake_X()
  - #Opener.fake_y()
  - #Opener.get_predictions()
  - #Opener.save_predictions()
  
  # Example
  
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
  
      def fake_X(self, n_samples):
          return []  # compute random fake data
  
      def fake_y(self, n_samples):
          return []  # compute random fake data
  
      def save_predictions(self, y_pred, path):
          with open(path, 'w') as fp:
              y_pred.to_csv(fp, index=False)
  
      def get_predictions(self, path):
          return pd.read_csv(path)
  ```
  
  # How to test locally an opener script
  
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
  
  Method resolution order:
      Opener
      abc.ABC
      builtins.object
  
  Methods defined here:
  
  fake_X(self, n_samples)
      Generate a fake matrix of features for offline testing.
      
      # Arguments
      
      n_samples (int): number of samples to return
      
      # Returns
      
      data: data labels object.
  
  fake_y(self, n_samples)
      Generate a fake target variable vector for offline testing.
      
      # Arguments
      
      n_samples (int): number of samples to return
      
      # Returns
      
      data: data labels object.
  
  get_X(self, folders)
      Load feature data from data sample folders.
      
      # Arguments
      
      folders: list of folders. Each folder represents a data sample.
      
      # Returns
      
      data: data object.
  
  get_predictions(self, path)
      Read file and return predictions vector.
      
      # Arguments
      
      path: string file path.
      
      # Returns
      
      predictions: predictions vector.
  
  get_y(self, folders)
      Load labels from data sample folders.
      
      # Arguments
      
      folders: list of folders. Each folder represents a data sample.
      
      # Returns
      
      data: data labels object.
  
  save_predictions(self, y_pred, path)
      Write predictions vector to file.
      
      # Arguments
      
      y_pred: predictions vector.
      path: string file path.
  
  ----------------------------------------------------------------------
  Data descriptors defined here:
  
  __dict__
      dictionary for instance variables (if defined)
  
  __weakref__
      list of weak references to the object (if defined)
  
  ----------------------------------------------------------------------
  Data and other attributes defined here:
  
  __abstractmethods__ = frozenset({'fake_X', 'fake_y', 'get_X', 'get_pre...

