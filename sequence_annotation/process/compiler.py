from abc import ABCMeta, abstractmethod, abstractproperty

class Compiler(metaclass=ABCMeta):
    @abstractmethod
    def process(self,model=None):
        pass
    
    def before_process(self,path=None):
        pass
    
    def after_process(self,path=None):
        pass
