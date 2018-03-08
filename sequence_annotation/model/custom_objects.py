from . import TruePositive
from . import TrueNegative
from . import FalsePositive
from . import FalseNegative
from . import SeqAnnModel
from . import SpecificTypeMetric
from . import AttrValidator
from . import CategoricalMetricFactory
class CustomObjectsFacade:
    def __init__(self, annotation_types, output_dim, terminal_signal,weights=None):
        self._annotation_types = annotation_types
        self._output_dim = output_dim
        self._terminal_signal = terminal_signal
        self._weights = weights
    def _create_metric(self,name,index):
        metric = SpecificTypeMetric(name,self._output_dim,index,self._terminal_signal)
        return metric
    @property
    def custom_objects(self):
        custom_objects = {}
        for index, ann_type in enumerate(self._annotation_types):
            custom_objects[ann_type+"_TP"] = TruePositive(ann_type,
                                                          self._create_metric(ann_type+"_TP",index))
            custom_objects[ann_type+"_TN"] = TrueNegative(ann_type,
                                                          self._create_metric(ann_type+"_TN",index))
            custom_objects[ann_type+"_FP"] = FalsePositive(ann_type,
                                                           self._create_metric(ann_type+"_FP",index))
            custom_objects[ann_type+"_FN"] = FalseNegative(ann_type,
                                                           self._create_metric(ann_type+"_FN",index))
        class_number = len(self._annotation_types)
        custom_objects['SeqAnnModel'] = SeqAnnModel
        factory = CategoricalMetricFactory()
        custom_objects["accuracy"] = factory.create("accuracy", "accuracy",class_number,
                                                    terminal_signal=self._terminal_signal)
        custom_objects["loss"] = factory.create("static_crossentropy", "loss",class_number,
                                                weights=self._weights,
                                                terminal_signal=self._terminal_signal)
        return custom_objects