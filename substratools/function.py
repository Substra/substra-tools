import inspect


def function(func):
    def wrap_and_call(*args, **kwargs):
        if "inputs" not in inspect.getargspec(func).args:
            raise

        if "outputs" not in inspect.getargspec(func).args:
            raise

        if "task_properties" not in inspect.getargspec(func).args:
            raise

        return func(*args, **kwargs)

    return wrap_and_call
