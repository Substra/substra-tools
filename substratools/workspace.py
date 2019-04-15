import os


class Workspace(object):
    """Filesystem workspace for algo execution."""
    LOG_FILENAME = 'log_model.log'

    def __init__(self, dirpath=None):
        self._root_path = dirpath if dirpath else os.getcwd()

        self._data_folder = os.path.join(self._root_path, 'data')
        self._pred_folder = os.path.join(self._root_path, 'pred')
        self._model_folder = os.path.join(self._root_path, 'model')

        paths = [self._data_folder, self._pred_folder, self._model_folder]
        for path in paths:
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

    @property
    def data_folder(self):
        return self._data_folder

    @property
    def pred_filepath(self):
        return os.path.join(self._pred_folder, 'pred')

    @property
    def score_filepath(self):
        return os.path.join(self._pred_folder, 'perf.json')

    @property
    def log_path(self):
        return os.path.join(self._model_folder, self.LOG_FILENAME)

    def save_model(self, buff, name='model'):
        with open(os.path.join(self._model_folder, name), 'w') as f:
            return f.write(buff)

    def load_model(self, name='model'):
        with open(os.path.join(self._model_folder, name), 'r') as f:
            return f.read()
