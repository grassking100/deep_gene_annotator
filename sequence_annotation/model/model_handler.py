from keras.models import load_model
from keras.optimizers import Adam
from . import CustomObjectsFacade
from . import ModelBuilder
class ModelHandler:
    @staticmethod
    def load_model(path):
        return load_model(path,compile=False)
    @staticmethod
    def build_model(model_setting):
        builder = ModelBuilder(model_setting)
        return builder.build()
    @classmethod
    def get_weights(cls,class_counts, method_name):
        if method_name=="reverse_counts":
            return cls._weights_reverse_count(class_counts,len(class_counts.keys()))
        else:
            mess = method_name+" is not implement yet."
            raise Exception(mess)
    @classmethod
    def _weights_reverse_count(cls,class_counts,scale):
        raw_weights = {}
        weights = {}
        for type_,count in class_counts.items():
            if count > 0:
                weight = 1 / count
            else:
                weight = 0
                scale -= 1
            raw_weights[type_] = weight
        sum_raw_weights = sum(raw_weights.values())
        for type_,count in raw_weights.items():
            weights[type_] = scale * count / sum_raw_weights
        return weights
    @staticmethod
    def compile_model(model,learning_rate,ann_types,loss_type,
                      values_to_ignore=None,weights=None,metric_types=None):
        weight_vec = None
        if weights is not None:
            weight_vec = []
            for type_ in ann_types:
                weight_vec.append(weights[type_])
        facade = CustomObjectsFacade(annotation_types = ann_types,
                                     values_to_ignore = values_to_ignore,
                                     weights = weight_vec,
                                     loss_type=loss_type,
                                     metric_types=metric_types)
        custom_objects = facade.custom_objects
        optimizer = Adam(lr=learning_rate)
        custom_metrics = []
        not_include_keys = ["loss"]
        for key,value in custom_objects.items():
            if key not in not_include_keys:
                custom_metrics.append(value)
        model.compile(optimizer=optimizer, loss=custom_objects['loss'],metrics=custom_metrics)