

class InvalidInterface(Exception):
    pass


class EmptyInterface(InvalidInterface):
    pass


class NotAFileError(Exception):
    pass


class MissingFileError(Exception):
    pass
