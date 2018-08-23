from abc import ABCMeta, abstractmethod
from ...model.custom_objects import CustomObjectsFacade
class Compiler(metaclass=ABCMeta):
    def __init__(self,optimizer,loss_type):
        self._optimizer = optimizer
        self._loss_type = loss_type
        self._record = {}
    @abstractmethod
    def process(self):
        pass
    @property
    def record(self):
        return self._record
    def before_process(self,path=None):
        pass
    def after_process(self,path=None):
        pass

class SimpleCompiler(Compiler):
    def process(self,model):
        model.compile(optimizer=self._optimizer, loss=self._loss_type)
    
class SeqAnnCompiler(Compiler):
    def __init__(self,optimizer,loss_type,ann_types=None,metric_types=None,values_to_ignore=None,
                 weights=None,dynamic_weight_method=None):
        super().__init__(optimizer,loss_type)
        self._record['ann_types'] = ann_types
        self._record['loss_type'] = loss_type
        self._record['metric_types'] = metric_types
        self._record['optimizer'] = optimizer
        self._record['values_to_ignore'] = values_to_ignore
        self._record['weights'] = weights
        self._record['dynamic_weight_method'] = dynamic_weight_method
        weight_vec = None
        if weights is not None:
            weight_vec = [weights[type_] for type_ in ann_types]
        self._facade = CustomObjectsFacade(annotation_types = ann_types,values_to_ignore = values_to_ignore,
                                           weights = weight_vec,loss_type=loss_type,metric_types=metric_types,
                                           dynamic_weight_method=dynamic_weight_method)
        self._optimizer = optimizer
    def process(self,model):
        custom_objects = self._facade.custom_objects
        custom_metrics = []
        not_include_keys = ["loss"]
        for key,value in custom_objects.items():
            if key not in not_include_keys:
                custom_metrics.append(value)
        model.compile(optimizer=self._optimizer, loss=custom_objects['loss'],metrics=custom_metrics)

class CompilerFactory:
    def create(self,type_):
        if type_ == 'simple':
            return SimpleCompiler
        elif type_ == 'seq_ann':
            return SeqAnnCompiler
        else:
            raise Exception(type_+' has not be supported yet.')