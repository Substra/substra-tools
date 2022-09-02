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
    ):
        super().__init__(dirpath=dirpath)

        assert input_data_folder_paths is None or isinstance(input_data_folder_paths, list)

        self.input_data_folder_paths = input_data_folder_paths or self._get_default_subpaths(
            DEFAULT_INPUT_DATA_FOLDER_PATH
        )

        for d in self.input_data_folder_paths:
            if d:
                makedir_safe(d)


class GenericAlgoWorkspace(OpenerWorkspace):
    """Filesystem workspace for user defined method execution."""

    def __init__(
        self,
        dirpath=None,
        log_path=None,
        chainkeys_path=None,
        inputs=None,
        outputs=None,
    ):
        super().__init__(dirpath, inputs.input_data_folder_paths) if inputs is not None else super().__init__(dirpath)

        self.log_path = log_path or self._get_default_path(DEFAULT_LOG_PATH)
        self.chainkeys_path = chainkeys_path or self._get_default_path(DEFAULT_CHAINKEYS_PATH)

        self.opener_path = inputs.opener_path if inputs is not None else None

        self.task_inputs = inputs.formatted_dynamic_resources if inputs is not None else None
        self.task_outputs = outputs.formatted_dynamic_resources if outputs is not None else None

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


class MetricsWorkspace(OpenerWorkspace):
    """Filesystem workspace for metrics execution."""

    def __init__(
        self,
        opener_path,
        dirpath=None,
        input_data_folder_paths=None,
        input_predictions_path=None,
        output_perf_path=None,
        log_path=None,
        chainkeys_path=None,
    ):
        super().__init__(
            dirpath=dirpath,
            input_data_folder_paths=input_data_folder_paths,
        )

        self.output_perf_path = output_perf_path or self._get_default_path(DEFAULT_OUTPUT_PERF_PATH)
        self.log_path = log_path or self._get_default_path(DEFAULT_LOG_PATH)
        self.chainkeys_path = chainkeys_path or self._get_default_path(DEFAULT_CHAINKEYS_PATH)
        self.input_predictions_path = input_predictions_path or self._get_default_path(DEFAULT_INPUT_PREDICTIONS_PATH)
        self.opener_path = opener_path

        dirs = []
        paths = [
            self.output_perf_path,
            self.log_path,
        ]
        dirs.extend([os.path.dirname(p) for p in paths])
        for d in dirs:
            if d:
                makedir_safe(d)


class AlgoWorkspace(GenericAlgoWorkspace):
    """Filesystem workspace for algo execution."""


class CompositeAlgoWorkspace(GenericAlgoWorkspace):
    """Filesystem workspace for composite algo execution."""


class AggregateAlgoWorkspace(GenericAlgoWorkspace):
    """Filesystem workspace for aggregate algo execution."""
