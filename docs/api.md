Help on class Algo in substratools:

substratools.Algo = class Algo(abc.ABC)
  Abstract base class for defining algo to run on the platform.
  
  To define a new algo script, subclass this class and implement the
  following abstract methods:
  
  - #Algo.train()
  - #Algo.predict()
  
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
      def train(self, inputs, outputs):
          new_model = None
          self.save_model(new_model, outputs["model"])
  
      def predict(self, inputs, outputs):
          model = self.load_model(inputs["model"])
          predictions = 0
          self.save_predictions(predictions, outputs["predictions"])
  
      def load_model(self, path):
          return json.load(path)
  
      def save_model(self, model, path):
          json.dump(model, path)
  
      def save_predictions(self, predictions, path):
          json.dump(predictions, path)
  
  if __name__ == '__main__':
      tools.algo.execute(DummyAlgo())
  ```
  
  # How to test locally an algo script
  
  # Using the command line
  
  The algo script can be directly tested through it's command line interface.
  For instance to train an algo using fake data, run the following command:
  
  ```sh
  python <script_path> train --fake-data --n-fake-samples 20 --log-level debug
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
  
  train_inputs={"X":X, "y":y, "model":None, "rank":0}
  train_outputs={"model":output_model_path}
  
  a.train(train_inputs, train_outputs)
  
  predict_inputs={"X":X, "model":input_model_path}
  predict_outputs={"predictions":output_predictions_path}
  
  a.predict(predict_inputs, predict_outputs)
  ```
  
  Method resolution order:
      Algo
      abc.ABC
      builtins.object
  
  Methods defined here:
  
  predict(self, inputs: substratools.algo.inputs, outputs: substratools.algo.outputs) -> None
      Get predictions from test data.
      
      This task corresponds to the creation of a testtuple on the Substra
      Platform.
      
      # Arguments
      
      inputs: TypedDict(
          "inputs",
          {
              InputIdentifiers.X: Any: testing data samples loaded with `Opener.get_X()`.
              InputIdentifiers.model: List[os.PathLike]: input model load with `Algo.load_model()` used for
                  predictions.
          },
      ),
      outputs: TypedDict(
          "outputs",
          {
              OutputIdentifiers.predictions: os.PathLike: output predictions path to save the predictions.
          },
      )
  
  train(self, inputs: substratools.algo.inputs, outputs: substratools.algo.outputs) -> None
      Train model and produce new model from train data.
      
      This task corresponds to the creation of a traintuple on the Substra
      Platform.
      
      # Arguments
      
      inputs: TypedDict(
          "inputs",
          {
              InputIdentifiers.X: List[Any]: training data samples loaded with `Opener.get_X()`.
              InputIdentifiers.y: List[Any]: training data samples labels loaded with `Opener.get_y()`.
              InputIdentifiers.models: Optional[
                  os.PathLike
              ]: list or generator of models loaded with `Algo.load_model()`.
              InputIdentifiers.rank: int: rank of the training task.
          },
      outputs: TypedDict(
          "outputs", {OutputIdentifiers.model: os.PathLike}: output model path to save the model.
      )
  
  ----------------------------------------------------------------------
  Data descriptors defined here:
  
  __dict__
      dictionary for instance variables (if defined)
  
  __weakref__
      list of weak references to the object (if defined)
  
  ----------------------------------------------------------------------
  Data and other attributes defined here:
  
  __abstractmethods__ = frozenset({'predict', 'train'})
  
  chainkeys_path = None
  
  compute_plan_path = None

Help on class CompositeAlgo in substratools:

substratools.CompositeAlgo = class CompositeAlgo(abc.ABC)
  Abstract base class for defining a composite algo to run on the platform.
  
  To define a new composite algo script, subclass this class and implement the
  following abstract methods:
  
  - #CompositeAlgo.train()
  - #CompositeAlgo.predict()
  
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
      def train(self, inputs, outputs):
          new_head_model = None
          new_trunk_model = None
          self.save_model(new_head_model, outputs["local"])
          self.save_model(new_trunk_model, outputs["shared"])
  
      def predict(self, inputs, outputs):
          predictions = 0
          self.save_predictions(predictions, outputs["predictions”])
  
      def load_model(self, path):
          return json.load(path)
  
      def save_model(self, model, path):
          json.dump(model, path)
  
      def save_predictions(self, predictions, path):
          json.dump(predictions, path)
  
  
  if __name__ == '__main__':
      tools.algo.execute(DummyCompositeAlgo())
  ```
  
  # How to test locally a composite algo script
  
  # Using the command line
  
  The composite algo script can be directly tested through it's command line interface.
  For instance to train an algo using fake data, run the following command:
  
  ```sh
  python <script_path> train --fake-data --n-fake-samples 20 --log-level debug
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
  inputs_train = {"X":X, "y":y, "local":None, "shared":None, "rank":0}
  outputs_train = {"local":head_model_path, "shared":trunk_model_path}
  head_model, trunk_model = a.train(inputs_train, outputs_train)
  
  inputs_predict = {"X":X, "local":None, "shared":None}
  outputs_predict = {"predictions":predictions_path}
  y_pred = a.predict(inputs_predict, outputs_predict)
  ```
  
  Method resolution order:
      CompositeAlgo
      abc.ABC
      builtins.object
  
  Methods defined here:
  
  predict(self, inputs: substratools.algo.inputs, outputs: substratools.algo.outputs) -> None
      Get predictions from test data.
      
      This task corresponds to the creation of a composite testtuple on the Substra
      Platform.
      
      # Arguments
      
      inputs: TypedDict(
          "inputs",
          {
              InputIdentifiers.X: Any: testing data samples loaded with `Opener.get_X()`.
              InputIdentifiers.local: os.PathLike: head model loaded with `CompositeAlgo.load_head_model()`.
              InputIdentifiers.shared: os.PathLike: trunk model loaded with `CompositeAlgo.load_trunk_model()`.
          },
      ),
      outputs: TypedDict(
          "outputs",
          {
              OutputIdentifiers.predictions: os.PathLike: output predictions path to save the predictions.
          },
      )
  
  train(self, inputs: substratools.algo.inputs, outputs: substratools.algo.outputs) -> None
      Train model and produce new composite models from train data.
      
      This task corresponds to the creation of a composite traintuple on the Substra
      Platform.
      
      # Arguments
      
      inputs: TypedDict(
          "inputs",
          {
              InputIdentifiers.X: Any: training data samples loaded with `Opener.get_X()`.
              InputIdentifiers.y: Any: training data samples labels loaded with `Opener.get_y()`.
              InputIdentifiers.local: Optional[os.PathLike]: head model loaded with `CompositeAlgo.load_head_model()`
                  (may be None).
              InputIdentifiers.shared: Optional[os.PathLike]: trunk model loaded with
                  `CompositeAlgo.load_trunk_model()` (may be None).
              InputIdentifiers.rank: int: rank of the training task.
          },
      ),
      outputs: TypedDict(
          "outputs",
          {
              OutputIdentifiers.local: os.PathLike: output head model path to save the head model.
              OutputIdentifiers.shared: os.PathLike: output trunk model path to save the trunk model.
          }
  
  ----------------------------------------------------------------------
  Data descriptors defined here:
  
  __dict__
      dictionary for instance variables (if defined)
  
  __weakref__
      list of weak references to the object (if defined)
  
  ----------------------------------------------------------------------
  Data and other attributes defined here:
  
  __abstractmethods__ = frozenset({'predict', 'train'})
  
  chainkeys_path = None
  
  compute_plan_path = None

Help on class AggregateAlgo in substratools:

substratools.AggregateAlgo = class AggregateAlgo(abc.ABC)
  Abstract base class for defining an aggregate algo to run on the platform.
  
  To define a new aggregate algo script, subclass this class and implement the
  following abstract methods:
  
  - #AggregateAlgo.aggregate()
  - #AggregateAlgo.predict()
  
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
      def aggregate(self, inputs, outputs):
          new_model = None
          self.save_model(outputs["model"])
  
      def predict(self, inputs, outputs):
          predictions = 0
          self.save_predictions(predictions, outputs["predictions”])
  
      def load_model(self, path):
          return json.load(path)
  
      def save_model(self, model, path):
          json.dump(model, path)
  
      def save_predictions(self, predictions, path):
          json.dump(predictions, path)
  
  
  if __name__ == '__main__':
      tools.algo.execute(DummyAggregateAlgo())
  ```
  
  # How to test locally an aggregate algo script
  
  # Using the command line
  
  The aggregate algo script can be directly tested through it's command line interface.
  For instance to train an algo using fake data, run the following command:
  
  ```sh
  python <script_path> aggregate --models_path <models_path> --models <model_name> --model <model_name>     --log-level debug
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
  
  aggregate(self, inputs: substratools.algo.inputs, outputs: substratools.algo.outputs)
      Aggregate models and produce a new model.
      
      This task corresponds to the creation of an aggregate tuple on the Substra
      Platform.
      
      # Arguments
      
      inputs: TypedDict(
          "inputs",
          {
              InputIdentifiers.models: List[os.PathLike]: list of models path loaded with `AggregateAlgo.load_model()`
              InputIdentifiers.rank: int: rank of the aggregate task.
          },
      ),
      outputs: TypedDict("outputs", {OutputIdentifiers.model: os.PathLike}): output model path to save the aggregated
          model.
  
  predict(self, inputs: substratools.algo.inputs, outputs: substratools.algo.outputs)
      Get predictions from test data.
      
      This task corresponds to the creation of a testtuple on the Substra
      Platform.
      
      # Arguments
      
      inputs: TypedDict(
          "inputs",
          {
              InputIdentifiers.X: Any: testing data samples loaded with `Opener.get_X()`.
              InputIdentifiers.model: os.PathLike: input model load with `AggregateAlgo.load_model()` used for
              predictions.
          },
      ),
      outputs: TypedDict("outputs", {"model": os.PathLike}): output predictions path to save the predictions.
  
  ----------------------------------------------------------------------
  Data descriptors defined here:
  
  __dict__
      dictionary for instance variables (if defined)
  
  __weakref__
      list of weak references to the object (if defined)
  
  ----------------------------------------------------------------------
  Data and other attributes defined here:
  
  __abstractmethods__ = frozenset({'aggregate', 'predict'})
  
  chainkeys_path = None
  
  compute_plan_path = None

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
      def score(self, inputs, outputs):
          y_true = inputs["y"]
          y_pred = self.load_predictions(inputs["predictions"])
          perf = accuracy_score(y_true, y_pred)
          tools.save_performance(perf, outputs["performance"])
  
      def load_predictions(self, predictions_path):
          return json.load(predictions_path)
  
  if __name__ == '__main__':
       tools.metrics.execute(AccuracyMetrics())
  ```
  
  # How to test locally a metrics script
  
  # Using the command line
  
  The metrics script can be directly tested through it's command line
  interface.  For instance to get the metrics from fake data, run the
  following command:
  
  ```sh
  python <script_path> --fake-data --n-fake-samples 20 --log-level debug
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
  
  inputs = {"y": y_true, "predictions":predictions_path}
  outputs = {"performance": performance_path}
  m.score(inputs, outputs)
  ```
  
  Method resolution order:
      Metrics
      abc.ABC
      builtins.object
  
  Methods defined here:
  
  score(self, inputs: substratools.metrics.inputs, outputs: substratools.metrics.outputs)
      Compute model perf from actual and predicted values.
      
      # Arguments
      
      inputs: TypedDict(
          "inputs",
          {
              InputIdentifiers.y: Any: actual values.
              InputIdentifiers.predictions: Any: path to predicted values.
          }
      ),
      outputs: TypedDict(
          "outputs",
          {
              OutputIdentifiers.performance: os.PathLike: path to save the performance of the model.
          }
      )
  
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
  
  get_y(self, folders)
      Load labels from data sample folders.
      
      # Arguments
      
      folders: list of folders. Each folder represents a data sample.
      
      # Returns
      
      data: data labels object.
  
  ----------------------------------------------------------------------
  Data descriptors defined here:
  
  __dict__
      dictionary for instance variables (if defined)
  
  __weakref__
      list of weak references to the object (if defined)
  
  ----------------------------------------------------------------------
  Data and other attributes defined here:
  
  __abstractmethods__ = frozenset({'fake_X', 'fake_y', 'get_X', 'get_y'}...

