"""This submodule provide abstract class about builder"""
from abc import ABCMeta, abstractmethod
class Builder(metaclass=ABCMeta):
    """The class provide api for building object"""
    def _validate(self):
        """Validate if all attribute is set correctly"""
        attrs = [attr for attr in dir(self) if attr.startswith('_Builder__')]
        for attr in attrs:
            if getattr(self, attr) is None:
                raise Exception("Builder needs "+attr+" to complete the quest")
    @abstractmethod
    def build(self):
        """Method to build object"""
        pass
