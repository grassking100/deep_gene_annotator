from keras.models import load_model
from keras.optimizers import Adam
from . import CustomObjectsFacade
from . import ModelBuilder
class ModelHandler():
    @staticmethod
    def load_model(path):
        return load_model(path,compile=False)
    @staticmethod
    def build_model(model_setting):
        builder = ModelBuilder(model_setting)
        return builder.build()
    @staticmethod
    def compile_model(model,learning_rate,ann_types,values_to_ignore=None,
                      class_weight=None,metric_types=None,loss_type=None,
                      dynamic_weight_method=None):
        weight_vec = None
        if class_weight is not None:
            weight_vec = []
            for type_ in ann_types:
                weight_vec.append(class_weight[type_])
        facade = CustomObjectsFacade(annotation_types = ann_types,
                                     values_to_ignore = values_to_ignore,
                                     weight = weight_vec,
                                     loss_type=loss_type,
                                     metric_types=metric_types,
                                     dynamic_weight_method=dynamic_weight_method)
        custom_objects = facade.custom_objects
        optimizer = Adam(lr=learning_rate)
        custom_metrics = []
        not_include_keys = ["loss"]
        for key,value in custom_objects.items():
            if key not in not_include_keys:
                custom_metrics.append(value)
        model.compile(optimizer=optimizer, loss=custom_objects['loss'],metrics=custom_metrics)