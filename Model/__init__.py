import tensorflow
import random
from sequence_annotation.Fasta_handler.Fasta_handler import *
from keras.layers import Input,Dropout,Convolution1D,Flatten,MaxPooling1D,LSTM,Reshape,Activation
import keras
from keras.models import Model
from sequence_annotation.Model import Model_build_helper
from sequence_annotation.Model.Sequence_annotation_model_factory import Sequence_annotation_model_factory
from sequence_annotation.Model.Model_trainer import Model_trainer
import numpy



