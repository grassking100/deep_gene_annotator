from . import Builder
from . import PrecisionFactory
from . import RecallFactory
from . import CategoricalAccuracyFactory
from . import CategoricalCrossEntropyFactory
from . import SeqAnnModel
class CustomObjectsBuilder(Builder):
    """This class create and stored custom objects list"""
    def __init__(self):
        super().__init__()
        self.__custom_objects = {}
        self.__loss_name = None
        self.__accuracy_name = None
    def add_precision_metric(self, precision_name, precision_function):
        """add precision metric"""
        self.__custom_objects[precision_name] = precision_function
        return self
    def add_recall_metric(self, recall_name, recall_function):
        """add recall metric"""
        self.__custom_objects[recall_name] = recall_function
        return self
    def add_loss_metric(self, loss_name, loss_function):
        """add loss function"""
        self.__loss_name = loss_name
        self.__custom_objects[loss_name] = loss_function
        return self
    def add_accuracy_metric(self, accuracy_name, accuracy_function):
        """add static accuracy"""
        self.__accuracy_name = accuracy_name
        self.__custom_objects[accuracy_name] = accuracy_function
        return self
    def add_model(self, model_name, model_function):
        """add model function"""
        self.__custom_objects[model_name] = model_function
        return self
    def _validate(self):
        """Validate if all attribute is set correctly"""
        keys = [self.__loss_name, self.__accuracy_name]
        for key in keys:
            if key not in self.__custom_objects.keys():
                raise Exception("Builder needs "+key+" to complete the quest")
    def build(self):
        """return custom objects dictionary"""
        self._validate()
        return self.__custom_objects
class CustomObjectsFacade:
    def __init__(self, annotation_types, output_dim, terminal_signal, accuracy_name, loss_name):
        self.__annotation_types = annotation_types
        self.__output_dim = output_dim
        self.__terminal_signal = terminal_signal
        self.__accuracy_name = accuracy_name
        self.__loss_name = loss_name
    @property
    def custom_objects(self):
        builder = CustomObjectsBuilder()
        for index, ann_type in enumerate(self.__annotation_types):
            precision_metric = PrecisionFactory("precision_"+ann_type,
                                                self.__output_dim, index,
                                                self.__terminal_signal).precision
            builder.add_precision_metric("precision_"+ann_type, precision_metric)
            recall_metric = RecallFactory("recall_"+ann_type,
                                          self.__output_dim, index,
                                          self.__terminal_signal).recall
            builder.add_recall_metric("recall_"+ann_type, recall_metric)
        class_number = len(self.__annotation_types)
        builder.add_model('SeqAnnModel', SeqAnnModel)
        accuracy = CategoricalAccuracyFactory(class_number).accuracy
        loss = CategoricalCrossEntropyFactory(class_number, True).cross_entropy
        builder.add_accuracy_metric(self.__accuracy_name,accuracy)
        builder.add_loss_metric(self.__loss_name,loss)
        return builder.build()
