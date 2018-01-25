"""This module includes librarys to help,build,train the model"""
import warnings
import tensorflow
import numpy
import matplotlib.pyplot as plt
from keras.callbacks import BaseLogger
import keras.callbacks as callbacks
import keras.optimizers as optimizers
import keras.losses as losses
import keras.backend as backend
from keras.engine.training import _collect_metrics
from keras.engine.training import _make_batches, _slice_arrays
from keras.layers import Input, Dropout, Convolution1D, Flatten
from keras.layers import MaxPooling1D, LSTM, Reshape, Activation
from keras.models import Model
from ..data_handler.training_data_handler import removed_terminal_tensors
from ..utils.container import Container
from ..utils.builder import Builder
from .seq_ann_model import SeqAnnModel
from .model_build_helper import CnnSettingBuilder
from .model_build_helper import CategoricalAccuracyFactory, CategoricalCrossEntropyFactory
from .model_build_helper import PrecisionFactory, RecallFactory
from .seq_ann_model_builder import SeqAnnModelBuilder
from .custom_objects import CustomObjectsFacade
