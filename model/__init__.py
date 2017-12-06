import tensorflow
import warnings
from keras.callbacks import BaseLogger
import keras.callbacks as callbacks
import keras.optimizers as optimizers
import keras.losses as losses
import keras.backend as backend
from keras.engine.training import _collect_metrics
from keras.engine.training import _make_batches,_slice_arrays
from sequence_annotation.data_handler.training_helper import *
from sequence_annotation.data_handler.fasta_handler import *
from keras.layers import Input,Dropout,Convolution1D,Flatten,MaxPooling1D,LSTM,Reshape,Activation
import keras
from keras.models import Model
from sequence_annotation.model.seq_ann_model import SeqAnnModel
from sequence_annotation.model.model_build_helper import CnnSettingCreator,categorical_accuracy_factory,categorical_crossentropy_factory,precision_creator,recall_creator,rename
from sequence_annotation.model.sequence_annotation_model_factory import SeqAnnModelFactory
from sequence_annotation.model.model_trainer import ModelTrainer
import numpy




