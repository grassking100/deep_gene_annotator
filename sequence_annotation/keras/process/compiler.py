from ...process.compiler import Compiler
from ..function.metric_builder import MetricBuilder
from ...utils.utils import create_folder

class SimpleCompiler(Compiler):
    def __init__(self,optimizer,loss_type,values_to_ignore=None,
                 weights=None,dynamic_weight_method=None,metrics=None):
        super().__init__()
        self._optimizer = optimizer
        self._loss_type = loss_type
        self._record['optimizer']=optimizer
        self._record['loss_type']=loss_type
        self._record['metrics'] = metrics
        self._record['values_to_ignore'] = values_to_ignore
        self._record['weights'] = weights
        self._record['dynamic_weight_method'] = dynamic_weight_method
        self._builder = MetricBuilder(values_to_ignore = values_to_ignore)
        self._builder.add_loss(type_=loss_type,weights=weights,
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

class AnnSeqCompiler(SimpleCompiler):

    def __init__(self,optimizer,loss_type,acc_type="categorical_accuracy",values_to_ignore=None,
                 weights=None,dynamic_weight_method=None,
                 ann_types=None,metrics=None):
        super().__init__(optimizer,loss_type,values_to_ignore,
                         weights,dynamic_weight_method,metrics)
        self._builder.add_accuracy(type_=acc_type)
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