from . import StatefulMetricFactory
from .metric import BatchCounter
from .metric import Accuracy
from .loss import Loss

class CustomObjectsFacade:
    def __init__(self, annotation_types,values_to_ignore=None,
                 weights=None, metric_types=None,loss_type=None,
                 dynamic_weight_method=None):
        self._annotation_types = annotation_types
        self._dynamic_weight_method = dynamic_weight_method
        self._values_to_ignore = values_to_ignore
        self._weights = weights
        self._metric_types = metric_types or ['TP','TN','FP','FN','accuracy','batch_counter']
        self._metric_types.append("loss")
        self._loss_type = loss_type
    @property
    def custom_objects(self):
        custom_objects = {}
        stateful_metric_factory = StatefulMetricFactory()
        if 'batch_counter' in self._metric_types:
            metric = BatchCounter("batch_counter")
            custom_objects["batch_counter"] = stateful_metric_factory.create(method_type='Constant',metric=metric)
        if self._annotation_types is not None:
            for index, ann_type in enumerate(self._annotation_types):
                for metric_type in self._metric_types:
                    if metric_type in ['TP','TN','FP','FN']:
                        name = "{ann_type}_{status}".format(ann_type=ann_type,status=metric_type)
                        metric = SpecificTypeMetric(name,values_to_ignore=self._values_to_ignore,
                                                    target_index=index)
                        metric_layer= stateful_metric_factory.create(metric_type,metric=metric,
                                                                  class_type=ann_type)
                        custom_objects[metric_layer.name]=metric_layer

        if 'accuracy' in self._metric_types:
            acc = Accuracy(name="accuracy",values_to_ignore=self._values_to_ignore)
            custom_objects["accuracy"]=acc
        loss = Loss(name="loss",weights=self._weights,
                    values_to_ignore=self._values_to_ignore,
                    type_=self._loss_type,
                    dynamic_weight_method=self._dynamic_weight_method)
        custom_objects["loss"]=loss
        return custom_objects