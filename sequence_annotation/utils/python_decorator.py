from .exception import ReturnNoneException
def validate_return(solution):
    def wrap(func):
        def validate(self=None):
            result = func(self)
            if result is None:
                raise ReturnNoneException(func.__name__,solution)
            return result
        return validate
    return wrap

def rename(newname):
    """rename the function"""
    def decorator(function):
        """rename input function name"""
        function.__name__ = newname
        return function
    return decorator