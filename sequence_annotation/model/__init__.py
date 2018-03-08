"""This module includes librarys to help,build,train the model"""
#import keras.optimizers as optimizers
#import keras.losses as losses
#from keras.engine.training import _collect_metrics
from ..data_handler.training_data_handler import removed_terminal_tensors
from ..utils.builder import Builder
from ..utils.validator import DataValidator, DictValidator, AttrValidator
from ..utils.exception import ReturnNoneException,NotPositiveException
from ..utils.python_decorator import rename
from .data_generator import DataGenerator
from .callback import ResultHistory
from .model_worker import ModelWorker
from .seq_ann_model import SeqAnnModel
from .metric import SpecificTypeMetric, CategoricalMetricFactory
from .model_build_helper import CnnSettingBuilder
from .metric_layer import TruePositive, TrueNegative, FalsePositive, FalseNegative
from .model_build_helper import CategoricalAccuracyFactory, CategoricalCrossEntropyFactory
from .model_build_helper import PrecisionFactory, RecallFactory
from .seq_ann_model_builder import SeqAnnModelBuilder
from .custom_objects import CustomObjectsFacade




