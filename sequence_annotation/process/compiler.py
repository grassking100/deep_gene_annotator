from abc import ABCMeta, abstractmethod, abstractproperty

class Compiler(metaclass=ABCMeta):
    def __init__(self,*args,**kwargs):
        self._record = {}

    @abstractmethod
    def process(self,model=None):
        pass
    
    @property
    def record(self):
        return self._record
    
    def before_process(self,path=None):
        pass
    
    def after_process(self,path=None):
        pass
    

