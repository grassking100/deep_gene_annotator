from . import Builder
from . import PrecisionFactory
from . import RecallFactory
from . import CategoricalAccuracyFactory
from . import CategoricalCrossEntropyFactory
from . import SeqAnnModel
from . import AttrValidator
class CustomObjectsBuilder(Builder):
    """This class create and stored custom objects list"""
    def __init__(self):
        super().__init__()
        self._custom_objects = {}
        self._loss_name = None
        self._accuracy_name = None
    def add_precision_metric(self, precision_name, precision_function):
        """add precision metric"""
        self._custom_objects[precision_name] = precision_function
        return self
    def add_recall_metric(self, recall_name, recall_function):
        """add recall metric"""
        self._custom_objects[recall_name] = recall_function
        return self
    def add_loss_metric(self, loss_name, loss_function):
        """add loss function"""
        self._loss_name = loss_name
        self._custom_objects[loss_name] = loss_function
        return self
    def add_accuracy_metric(self, accuracy_name, accuracy_function):
        """add static accuracy"""
        self._accuracy_name = accuracy_name
        self._custom_objects[accuracy_name] = accuracy_function
        return self
    def add_model(self, model_name, model_function):
        """add model function"""
        self._custom_objects[model_name] = model_function
        return self
    def _validate(self):
        attr_validator = AttrValidator(self,False,True,False,None)
        attr_validator.validate()
    def build(self):
        """return custom objects dictionary"""
        self._validate()
        return self._custom_objects
class CustomObjectsFacade:
    def __init__(self, annotation_types, output_dim, terminal_signal,weights=None):
        self._annotation_types = annotation_types
        self._output_dim = output_dim
        self._terminal_signal = terminal_signal
        self._weights = weights
    @property
    def custom_objects(self):
        builder = CustomObjectsBuilder()
        for index, ann_type in enumerate(self._annotation_types):
            precision_metric = PrecisionFactory("precision_"+ann_type,
                                                self._output_dim, index,
                                                self._terminal_signal).precision
            builder.add_precision_metric("precision_"+ann_type, precision_metric)
            recall_metric = RecallFactory("recall_"+ann_type,
                                          self._output_dim, index,
                                          self._terminal_signal).recall
            builder.add_recall_metric("recall_"+ann_type, recall_metric)
        class_number = len(self._annotation_types)
        builder.add_model('SeqAnnModel', SeqAnnModel)
        accuracy = CategoricalAccuracyFactory(class_number,self._terminal_signal).accuracy
        loss = CategoricalCrossEntropyFactory(class_number,
                                              True, self._weights,
                                              self._terminal_signal).cross_entropy
        builder.add_accuracy_metric("accuracy",accuracy)
        builder.add_loss_metric("loss",loss)
        return builder.build()
