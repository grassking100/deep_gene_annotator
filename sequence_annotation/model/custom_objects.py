from . import MetricFactory
from . import MetricLayerFactory
from . import CategoricalCrossentropyFactory
from . import CategoricalAccuracyFactory

class CustomObjectsFacade:
    def __init__(self, annotation_types,values_to_ignore=None,
                 weights=None, metric_types=None):
        self._annotation_types = annotation_types
        self._values_to_ignore = values_to_ignore
        self._weights = weights
        self._metric_types = metric_types or ['TP','TN','FP','FN']
        #self._metric_types = metric_types or ['TP','TN','FP','FN','RL','PL']
    @property
    def custom_objects(self):
        custom_objects = {}
        metric_factory = MetricFactory()
        loss_factory = CategoricalCrossentropyFactory()
        acc_factory = CategoricalAccuracyFactory()
        metric_layer_factory = MetricLayerFactory()
        metric = metric_factory.create(type_="dependent", name="constant")
        custom_objects["Constant"] = metric_layer_factory.create("Constant", metric=metric)
        for index, ann_type in enumerate(self._annotation_types):
            for metric_type in self._metric_types:
                name = "{ann_type}_{status}".format(ann_type=ann_type,status=metric_type)
                metric = metric_factory.create("specific_type",name,
                                               values_to_ignore=self._values_to_ignore,
                                               target_index=index)
                metric_layer= metric_layer_factory.create(metric_type,metric=metric,
                                                          class_type=ann_type)
                #custom_objects[metric_layer.name]=metric_layer
        acc = acc_factory.create(name="accuracy",values_to_ignore=self._values_to_ignore)
        loss = loss_factory.create(name="loss",weights=self._weights,
                                   values_to_ignore=self._values_to_ignore)
        #custom_objects["accuracy"]=acc
        custom_objects["loss"]=loss
        return custom_objects