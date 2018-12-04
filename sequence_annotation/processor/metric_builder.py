from .stateful_metric import StatefulMetric
from .metric import TP,FP,TN,FN
from .metric import BatchCount
from .metric import Accuracy
from .loss import Loss

class MetricBuilder():
    def __init__(self,values_to_ignore):
        self._custom_objects = {}
        self._values_to_ignore = values_to_ignore

    def add_loss(self,*args,**kwargs):
        loss = Loss(*args,**kwargs)
        self._custom_objects["loss"]=loss
        return self

    def add_accuracy(self,*args,**kwargs):
        acc = Accuracy(*args,**kwargs)
        self._custom_objects["accuracy"]=acc
        return self

    def add_batch_count(self):
        metric = BatchCount(*args,**kwargs)
        stateful_metric = StatefulMetric(metric)
        self._custom_objects[stateful_metric.name] = stateful_metric
        return self

    def _add_binary_stateful_metric(self,class_type,name,ann_type,ann_types):
        index = ann_types.index(ann_type)
        metric = class_type(index,name,values_to_ignore=self._values_to_ignore)
        stateful_metric = StatefulMetric(metric)
        self._custom_objects[stateful_metric.name] = stateful_metric

    def add_TP(self,ann_type,ann_types,name=None):
        name = name or ann_type+"_TP"
        self._add_binary_stateful_metric(TP,name,ann_type,ann_types)
        return self

    def add_TN(self,ann_type,ann_types,name=None):
        name = name or ann_type+"_TN"
        self._add_binary_stateful_metric(TN,name,ann_type,ann_types)
        return self

    def add_FP(self,ann_type,ann_types,name=None):
        name = name or ann_type+"_FP"
        self._add_binary_stateful_metric(FP,name,ann_type,ann_types)
        return self

    def add_FN(self,ann_type,ann_types,name=None):
        name = name or ann_type+"_FN"
        self._add_binary_stateful_metric(FN,name,ann_type,ann_types)
        return self

    def build(self):
        return self._custom_objects
