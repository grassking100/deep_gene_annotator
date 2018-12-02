import os
import sys
from os.path import abspath, expanduser
import unittest
from sequence_annotation.model.model_processor import SimpleModel,ModelCreator
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.processor.compiler import SimpleCompiler,AnnSeqCompiler
from sequence_annotation.processor.stateful_metric import StatefulMetric
from sequence_annotation.processor.metric import BatchCount,SampleCount
from sequence_annotation.processor.data_processor import AnnSeqData,SimpleData
from sequence_annotation.worker.train_worker import TrainWorker
from sequence_annotation.worker.test_worker import TestWorker
from sequence_annotation.data_handler.fasta import read_fasta
from sequence_annotation.data_handler.json import read_json
from sequence_annotation.data_handler.seq_converter import SeqConverter
from sequence_annotation.pipeline.wrapper import fit_generator_wrapper_generator,fit_wrapper_generator
from sequence_annotation.pipeline.wrapper import evaluate_generator_wrapper_generator
from sequence_annotation.pipeline.pipeline import Pipeline
from sequence_annotation.utils.utils import model_method
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation,Input,RNN,Convolution1D
from keras.models import Model
import keras
from sequence_annotation.genome_handler.ann_seq_processor import class_count

from keras.engine.topology import Layer
from abc import ABCMeta
from keras import backend as K
import tensorflow as tf
class MaskedCNN(Convolution1D):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.supports_masking = True
    def compute_mask(self, input_, input_mask=None):
        # do not pass the mask to the next layers
        return input_mask
class Masking(Layer):
    def __init__(self, mask_value=0, **kwargs):
        super(Masking, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value

    def compute_mask(self, inputs, mask=None):
        output_mask = K.any(K.not_equal(inputs, self.mask_value), axis=-1)
        #a=tf.Print(output_mask,[output_mask], message='\noutput_mask = ',summarize=10)
        return output_mask

    def call(self, inputs):
        boolean_mask = self.compute_mask(inputs)
        return inputs#K.cast(boolean_mask, K.dtype(inputs))
class BatchCounter(Layer):
    def __init__(self, name="batch_counter", **kwargs):
        super(BatchCounter, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.batches = keras.backend.variable(value=0, dtype="float32")

    def reset_states(self):
        keras.backend.set_value(self.batches, 0)

    def __call__(self, y_true, y_pred):
        updates = [
            keras.backend.update_add(
                self.batches, 
                keras.backend.variable(value=1, dtype="float32"))]
        self.add_update(updates)
        return self.batches

class TestPipeline(unittest.TestCase):
    def test_binary_type_length(self):
        
        seq_data = np.load(abspath(expanduser(__file__+'/../data/small_answer.npy'))).item()
        seqs = AnnSeqContainer().from_dict(seq_data)
        ann_types = seqs.ANN_TYPES
        fasta = read_fasta(abspath(expanduser(__file__+'/../data/small_seq.fasta')))
    
        i = Input(shape=(3,4))
        m = Masking(3430)(i)
        #r=RNN(IRNNCell(5),return_sequences=True)(m)
        d=MaskedCNN(filters=2,kernel_size=1)(m)
        #r=Dense(5)(d)
        model_ = Model(inputs=i, outputs=d)
        model = SimpleModel(model_)
        custom_metrics=[BatchCounter()]#StatefulMetric(SampleCount()),StatefulMetric(BatchCount())]
        compiler = SimpleCompiler('adam','mse',metrics=custom_metrics)
        seq_converter = SeqConverter(codes="ATCG", with_soft_masked_status=False)
        fi=list(fasta.keys())
        data = AnnSeqData({'data':{'training':{'inputs':fasta,'answers':seqs}},
                           'ANN_TYPES':ann_types},
                           seq_converter = seq_converter)
        #print(fi)
        train_worker = TrainWorker(is_verbose_visible=False)
        train_wrapper = fit_wrapper_generator(batch_size=1,epochs=10,verbose=1)
        train_pipeline = Pipeline(model,data,compiler,train_worker,
                                  train_wrapper,is_prompt_visible=False)
        train_pipeline.execute()
        result = train_worker.result['batch_counter']
        #print(result)
        print(result==[3.0]*100)
        #print(model_method(model.model,0,1)([[[0,0,0,1,0,0],[1,0,0,0,0,0]]]))
        #print(model.model.evaluate(x=np.array(data.data['training']['inputs']),
        #                           y=np.array(data.data['training']['answers']),
        #                           batch_size=1
        #                                   ))
        #print(model_method(model.model,0,2)([data.data['training']['inputs']]))
        #print(model_method(model.model,0,3)([[[0,0,0,1,0,0],[1,0,0,0,0,0]]]))
        count = 0
        