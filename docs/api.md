# Algo
```python
Algo(self, /, *args, **kwargs)
```
Abstract base class for defining algo to run on the platform.
## train
```python
Algo.train(self, X, y, models, rank)
```
Train algorithm and produce new model.

Must return a tuple (predictions, model).

## predict
```python
Algo.predict(self, X, y, model)
```
Load model and save predictions made on train data.

Must return predictions.

## dry_run
```python
Algo.dry_run(self, X, y, models, rank)
```
Train model dry run mode.
## load_model
```python
Algo.load_model(self, path)
```
Load model from path.
## save_model
```python
Algo.save_model(self, model, path)
```
Save model to path.
# Metrics
```python
Metrics(self, /, *args, **kwargs)
```

## score
```python
Metrics.score(self, y_true, y_pred)
```
Returns the macro-average recall.

Must return a float.

# Opener
```python
Opener(self, /, *args, **kwargs)
```
Dataset opener abstract base class.
## get_X
```python
Opener.get_X(self, folders)
```
Load feature data from data sample folders.
## get_y
```python
Opener.get_y(self, folders)
```
Load labels from data sample folders.
## get_pred
```python
Opener.get_pred(self, path)
```
Get predictions from path.
## save_pred
```python
Opener.save_pred(self, y_pred, path)
```
Save predictions to path.
