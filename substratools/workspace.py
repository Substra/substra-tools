import os


def makedir_safe(path):
    """Create dir (no failure)."""
    try:
        os.makedirs(path)
    except (FileExistsError, PermissionError):
        pass


DEFAULT_INPUT_DATA_FOLDER_PATH = 'data/'
DEFAULT_INPUT_MODELS_FOLDER_PATH = 'model/'
DEFAULT_INPUT_PREDICTIONS_PATH = 'pred/pred'
DEFAULT_OUTPUT_MODEL_PATH = 'model/model'
DEFAULT_OUTPUT_PREDICTIONS_PATH = 'pred/pred'
DEFAULT_OUTPUT_PERF_PATH = 'pred/perf.json'
DEFAULT_LOG_PATH = 'model/log_model.log'


class Workspace(object):
    """Filesystem workspace for algo and metrics execution."""
    # TODO have different workspaces for algo train/predict and metrics

    def __init__(self,
                 dirpath=None,
                 input_data_folder_path=None,
                 input_models_folder_path=None,
                 input_predictions_path=None,
                 output_model_path=None,
                 output_predictions_path=None,
                 output_perf_path=None,
                 log_path=None, ):

        self._workdir = dirpath if dirpath else os.getcwd()

        def _get_default_path(path):
            return os.path.join(self._workdir, path)

        self.input_data_folder_path = input_data_folder_path or \
            _get_default_path(DEFAULT_INPUT_DATA_FOLDER_PATH)

        self.input_models_folder_path = input_models_folder_path or \
            _get_default_path(DEFAULT_INPUT_MODELS_FOLDER_PATH)

        self.input_predictions_path = input_predictions_path or \
            _get_default_path(DEFAULT_INPUT_PREDICTIONS_PATH)

        self.output_model_path = output_model_path or \
            _get_default_path(DEFAULT_OUTPUT_MODEL_PATH)

        self.output_predictions_path = output_predictions_path or \
            _get_default_path(DEFAULT_OUTPUT_PREDICTIONS_PATH)

        self.output_perf_path = output_perf_path or \
            _get_default_path(DEFAULT_OUTPUT_PERF_PATH)

        self.log_path = log_path or \
            _get_default_path(DEFAULT_LOG_PATH)

        dirs = [
            self.input_models_folder_path,
            self.input_data_folder_path,
        ]
        paths = [
            self.input_predictions_path,
            self.output_model_path,
            self.output_predictions_path,
            self.output_perf_path,
            self.log_path,
        ]
        dirs.extend([os.path.dirname(p) for p in paths])
        for d in dirs:
            if d:
                makedir_safe(d)
