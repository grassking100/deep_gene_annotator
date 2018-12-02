from abc import ABCMeta, abstractmethod
from .metric_builder import MetricBuilder
from ..utils.utils import create_folder

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
    def __init__(self,optimizer,loss_type,values_to_ignore=None,
                 weights=None,dynamic_weight_method=None,metrics=None):
        super().__init__(optimizer,loss_type,metrics)
        self._record['values_to_ignore'] = values_to_ignore
        self._record['weights'] = weights
        self._record['dynamic_weight_method'] = dynamic_weight_method
        self._builder = MetricBuilder(values_to_ignore = values_to_ignore)
        self._builder.add_loss(loss_type=loss_type,weights=weights,
                               dynamic_weight_method=dynamic_weight_method)
        self._loss = self._builder.build()['loss']
        self._optimizer = optimizer
        self._metrics = metrics or []
    def process(self,model):
        model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)
    def before_process(self,path=None):
        if path is not None:
            json_path = create_folder(path) + "/setting/compiler.json"
            with open(json_path,'w') as fp:
                json.dump(self._record,fp)

class AnnSeqCompiler(Compiler):
    def __init__(self,optimizer,loss_type,values_to_ignore=None,
                 weights=None,dynamic_weight_method=None,
                 ann_types=None,metrics=None):
        super().__init__(optimizer,loss_type,values_to_ignore,
                         weights,dynamic_weight_method,metrics)
        self._record['ann_types'] = ann_types
        for ann_type in ann_types:
            self._builder.add_TP(ann_type,ann_types)
            self._builder.add_FP(ann_type,ann_types)
            self._builder.add_TN(ann_type,ann_types)
            self._builder.add_FN(ann_type,ann_types)
        build_in_metrics = []
        not_include_keys = ["loss"]
        builded_metrics = self._builder.build()
        for key,value in builded_metrics.items():
            if key not in not_include_keys:
                build_in_metrics.append(value)
        self._metrics += build_in_metrics
