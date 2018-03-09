from . import SeqAnnModel
from . import CategoricalMetricFactory
from . import MetricLayerFactory
class CustomObjectsFacade:
    def __init__(self, annotation_types,terminal_signal,weights=None):
        self._annotation_types = annotation_types
        self._terminal_signal = terminal_signal
        self._weights = weights
    @property
    def custom_objects(self):
        custom_objects = {}
        categorical_metric_factory = CategoricalMetricFactory()
        metric_layer_factory = MetricLayerFactory()
        metric_types=['TP','TN','FP','FN']
        for index, ann_type in enumerate(self._annotation_types):
            for metric_type in metric_types:
                name = "{ann_type}_{status}".format(ann_type=ann_type,status=metric_type)
                metric = categorical_metric_factory.create("specific_type",name,
                                                           terminal_signal=self._terminal_signal,
                                                           target_index=index)
                custom_objects[name]= metric_layer_factory.create(metric_type,ann_type,metric)
        custom_objects['SeqAnnModel'] = SeqAnnModel
        custom_objects["accuracy"] = categorical_metric_factory.create("accuracy","accuracy",
                                                                       terminal_signal=self._terminal_signal)
        custom_objects["loss"] = categorical_metric_factory.create("static_crossentropy","loss",
                                                                   weights=self._weights,
                                                                   terminal_signal=self._terminal_signal)
        return custom_objects
