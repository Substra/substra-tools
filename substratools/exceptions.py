class InvalidInterfaceError(Exception):
    pass


class EmptyInterfaceError(InvalidInterfaceError):
    pass


class NotAFileError(Exception):
    pass


class MissingFileError(Exception):
    pass


class InvalidInputOutputsError(Exception):
    pass
