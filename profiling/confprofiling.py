"""Config Profiling."""
from os import path


def get_source(source_path=None, source_name=None):
    """Take a test file and pass it to a function.

    Args:
        source_path (str, optional): source folder path. Defaults to None.
        source_name (str, optional): source name. Defaults to None.
    """

    def decorator(function):
        def wrapper(*args, **kwargs):

            source = path.join("testdata", source_path, source_name)

            return function(*args, source=source, **kwargs)

        return wrapper

    return decorator
