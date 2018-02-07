from . import ReturnNoneException
def validate_return(solution):
    def wrap(func):
        def validate(self=None):
            result = func(self)
            if result is None:
                raise ReturnNoneException(func.__name__,solution)
            return result
        return validate
    return wrap

