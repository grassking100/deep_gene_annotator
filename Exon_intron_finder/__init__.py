import tensorflow
import random
from gene_prediction.Fasta_handler.Fasta_handler import *
from keras.layers import Input,Dropout,Convolution1D,Flatten,MaxPooling1D,LSTM,Reshape,Activation
import keras
from keras.models import Model
from gene_prediction.Exon_intron_finder.Exon_intron_finder import Exon_intron_finder_factory
from gene_prediction.Exon_intron_finder.Model_evaluator import Model_evaluator
import numpy

