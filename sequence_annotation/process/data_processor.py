from abc import ABCMeta, abstractmethod,abstractproperty

class IDataProcessor(metaclass=ABCMeta):
    @abstractmethod
    def process(self):
        pass
    @abstractproperty
    def record(self):
        pass
    @abstractproperty
    def data(self):
        pass
    @abstractmethod
    def before_process(self,path=None):
        pass
    @abstractmethod
    def after_process(self,path=None):
        pass

class SimpleData(IDataProcessor):
    def __init__(self,data):
        self._data = data
        self._record = {}
    @property
    def record(self):
        return self._record
    @property
    def data(self):
        return self._data
    def before_process(self,path=None):
        pass
    def after_process(self,path=None):
        pass
    def process(self):
        pass

