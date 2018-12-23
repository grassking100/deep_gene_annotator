from abc import ABCMeta, abstractmethod, abstractproperty
from ..metric.metric_builder import MetricBuilder
from ...utils.utils import create_folder

class Compiler(metaclass=ABCMeta):
    
    def __init__(self,optimizer,loss_type,metrics=None):
        self._optimizer = optimizer
        self._loss_type = loss_type
        self._metrics = metrics
        self._record = {'optimizer':optimizer,
                        'loss_type':loss_type,
                        'metrics':metrics
                        }
    
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
    

