import abc
import os


def makedir_safe(path):
    """Create dir (no failure)."""
    try:
        os.makedirs(path)
    except (FileExistsError, PermissionError):
        pass


DEFAULT_INPUT_DATA_FOLDER_PATH = "data/"
DEFAULT_INPUT_PREDICTIONS_PATH = "pred/pred"
DEFAULT_OUTPUT_PREDICTIONS_PATH = "pred/pred"
DEFAULT_OUTPUT_PERF_PATH = "pred/perf.json"
DEFAULT_LOG_PATH = "model/log_model.log"
DEFAULT_CHAINKEYS_PATH = "chainkeys/"


class Workspace(abc.ABC):
    """Filesystem workspace for task execution."""

    def __init__(self, dirpath=None):
        self._workdir = dirpath if dirpath else os.getcwd()

    def _get_default_path(self, path):
        return os.path.join(self._workdir, path)

    def _get_default_subpaths(self, path):
        rootpath = os.path.join(self._workdir, path)
        if os.path.isdir(rootpath):
            return [
                os.path.join(rootpath, subfolder)
                for subfolder in os.listdir(rootpath)
                if os.path.isdir(os.path.join(rootpath, subfolder))
            ]
        return []


class OpenerWorkspace(Workspace):
    """Filesystem workspace required by the opener."""

    def __init__(
        self,
        dirpath=None,
        input_data_folder_paths=None,
        input_predictions_path=None,
        output_predictions_path=None,
    ):
        super().__init__(dirpath=dirpath)

        assert input_data_folder_paths is None or isinstance(input_data_folder_paths, list)

        self.input_data_folder_paths = input_data_folder_paths or self._get_default_subpaths(
            DEFAULT_INPUT_DATA_FOLDER_PATH
        )

        self.input_predictions_path = input_predictions_path or self._get_default_path(DEFAULT_INPUT_PREDICTIONS_PATH)

        self.output_predictions_path = output_predictions_path or self._get_default_path(
            DEFAULT_OUTPUT_PREDICTIONS_PATH
        )

        dirs = []
        dirs.extend(self.input_data_folder_paths)
        paths = [
            self.input_predictions_path,
            self.output_predictions_path,
        ]
        dirs.extend([os.path.dirname(p) for p in paths])
        for d in dirs:
            if d:
                makedir_safe(d)


class MetricsWorkspace(OpenerWorkspace):
    """Filesystem workspace for metrics execution."""

    def __init__(
        self,
        dirpath=None,
        input_data_folder_paths=None,
        input_predictions_path=None,
        output_perf_path=None,
        log_path=None,
    ):
        super().__init__(
            dirpath=dirpath,
            input_data_folder_paths=input_data_folder_paths,
            input_predictions_path=input_predictions_path,
        )

        self.output_perf_path = output_perf_path or self._get_default_path(DEFAULT_OUTPUT_PERF_PATH)

        self.log_path = log_path or self._get_default_path(DEFAULT_LOG_PATH)

        dirs = []
        paths = [
            self.output_perf_path,
            self.log_path,
        ]
        dirs.extend([os.path.dirname(p) for p in paths])
        for d in dirs:
            if d:
                makedir_safe(d)


class AlgoWorkspace(OpenerWorkspace):
    """Filesystem workspace for algo execution."""

    def __init__(
        self,
        dirpath=None,
        input_data_folder_paths=None,
        input_model_paths=None,
        input_predictions_path=None,
        output_model_path=None,
        output_predictions_path=None,
        log_path=None,
        chainkeys_path=None,
    ):
        super().__init__(
            dirpath=dirpath,
            input_data_folder_paths=input_data_folder_paths,
            input_predictions_path=input_predictions_path,
            output_predictions_path=output_predictions_path,
        )

        self.input_model_paths = input_model_paths
        self.output_model_path = output_model_path

        self.log_path = log_path or self._get_default_path(DEFAULT_LOG_PATH)

        self.chainkeys_path = chainkeys_path or self._get_default_path(DEFAULT_CHAINKEYS_PATH)

        dirs = [
            self.chainkeys_path,
        ]
        paths = [
            self.log_path,
        ]

        dirs.extend([os.path.dirname(p) for p in paths])
        for d in dirs:
            if d:
                makedir_safe(d)


class CompositeAlgoWorkspace(OpenerWorkspace):
    def __init__(
        self,
        dirpath=None,
        input_data_folder_paths=None,
        input_predictions_path=None,
        input_head_model_path=None,
        input_trunk_model_path=None,
        output_head_model_path=None,
        output_trunk_model_path=None,
        output_predictions_path=None,
        log_path=None,
        chainkeys_path=None,
    ):
        super().__init__(
            dirpath=dirpath,
            input_data_folder_paths=input_data_folder_paths,
            input_predictions_path=input_predictions_path,
            output_predictions_path=output_predictions_path,
        )

        self.input_head_model_path = input_head_model_path
        self.input_trunk_model_path = input_trunk_model_path

        self.output_head_model_path = output_head_model_path
        self.output_trunk_model_path = output_trunk_model_path

        self.log_path = log_path or self._get_default_path(DEFAULT_LOG_PATH)

        self.chainkeys_path = chainkeys_path or self._get_default_path(DEFAULT_CHAINKEYS_PATH)

        dirs = [
            self.chainkeys_path,
        ]
        paths = [
            self.log_path,
        ]
        dirs.extend([os.path.dirname(p) for p in paths])
        for d in dirs:
            if d:
                makedir_safe(d)


class AggregateAlgoWorkspace(AlgoWorkspace):
    """Filesystem workspace for aggregate algo execution."""

    pass
