import json
import numpy as np
from sklearn.externals import joblib


class SubstraModel:
    """Base class for algo/models submitted/trained on the Substra platform"""
    def __init__(self, data_folder="./data", pred_folder="./pred", model_file="./model/model"):
        self.data_folder = data_folder
        self.pred_folder = pred_folder
        self.model_file = model_file

    def save_json_sklearn(self, model):
        """Save estimated params of a sklearn model in a json file.
        Does not work for sklearn.ensemble estimators"""
        attr = model.__dict__
        estimated_params = {key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in attr.items() if key[-1] == "_" and key != "loss_function_"}
        with open(self.model_file, 'w') as f:
            json.dump(estimated_params, f)

    def load_json_sklearn(self, model):
        """ Load estimated params of a trained sklearn model from a json file"""
        with open(self.model_file, 'r') as f:
            attr = json.load(f)
        for key, value in attr.items():
            if isinstance(value, list):
                setattr(model, key, np.array(value))
            else:
                setattr(model, key, value)

    def save_pkl_sklearn(self, model):
        """Save sklearn model in pickle file (for ensemble models).
        Not secure, may be modified later if we do really use such types of models"""
        joblib.dump(model, self.model_file)

    def load_pkl_sklearn(self):
        """Load sklearn model in pickle file (for ensemble models).
        Not secure, may be modified later if we do really use such types of models"""
        return joblib.load(self.model_file)
