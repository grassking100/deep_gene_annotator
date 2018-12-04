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
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation,Input,RNN
from keras.models import Model

class TestPipeline(unittest.TestCase):
    def test_simple_train(self):
        try:
            model = Sequential([
                Dense(32, input_shape=(2,)),
                Activation('relu'),
                Dense(2),
                Activation('softmax')
            ])
            simple_model = SimpleModel(model)
            compiler = SimpleCompiler('adam','binary_crossentropy')
            data = SimpleData({'training':{'inputs':[[1,0]],'answers':[[1,0]]}})
            worker = TrainWorker(is_verbose_visible=False)
            wrapper = fit_generator_wrapper_generator(verbose=0)
            pipeline = Pipeline(simple_model,data,compiler,worker,
                                wrapper,is_prompt_visible=False)

            pipeline.execute()
        except Exception as e:
            raise e
            self.fail("There are some unexpected exceptions occur.")

    def test_simple_test(self):
        try:
            model = Sequential([
                Dense(32, input_shape=(2,)),
                Activation('relu'),
                Dense(2),
                Activation('softmax')
            ])
            simple_model = SimpleModel(model)
            compiler = SimpleCompiler('adam','binary_crossentropy')
            data = SimpleData({'testing':{'inputs':[[1,0]],'answers':[[1,0]]}})
            worker = TestWorker(is_verbose_visible=False)
            wrapper = evaluate_generator_wrapper_generator(verbose=0)
            pipeline = Pipeline(simple_model,data,compiler,worker
                                ,wrapper,is_prompt_visible=False)
            pipeline.execute()
        except Exception as e:
            raise e
            self.fail("There are some unexpected exceptions occur.")

    def test_ann_seq_pipeline(self):
        try:
            ann_types = ["utr_5","utr_3","intron","cds","intergenic_region"]
            seq_data = np.load(abspath(expanduser(__file__+'/../../data/seqs.npy'))).item()
            seqs = AnnSeqContainer().from_dict(seq_data)
            fasta = read_fasta(abspath(expanduser(__file__+'/../../data/seqs.fasta')))
            model_setting_path = abspath(expanduser(__file__+'/../../setting/model_setting.json'))
            model_setting = read_json(model_setting_path)
            model = ModelCreator(model_setting)
            train_compiler = AnnSeqCompiler('adam','mse',ann_types=ann_types,
                                            metrics=[StatefulMetric(SampleCount()),
                                                     StatefulMetric(BatchCount())])
            seq_converter = SeqConverter(codes="ATCGN", with_soft_masked_status=True)
            data = AnnSeqData({'data':{'training':{'inputs':fasta,'answers':seqs}},
                               'ANN_TYPES':ann_types},
                               seq_converter = seq_converter)
            train_worker = TrainWorker(is_verbose_visible=False)
            train_wrapper = fit_generator_wrapper_generator(batch_size=1, epochs=30,verbose=0)
            train_pipeline = Pipeline(model,data,train_compiler,train_worker,
                                      train_wrapper,is_prompt_visible=False)
            train_pipeline.execute()
            test_worker = TestWorker(is_verbose_visible=False)
            test_data = AnnSeqData({'data':{'testing':{'inputs':fasta,'answers':seqs}},
                                    'ANN_TYPES':ann_types},
                                     seq_converter = seq_converter)
            simple_model = SimpleModel(model.model)
            evaluate_wrapper = evaluate_generator_wrapper_generator(verbose=0)
            test_pipeline = Pipeline(simple_model,test_data,None,test_worker,
                                     evaluate_wrapper,is_prompt_visible=False)
            test_pipeline.execute()
            self.assertEqual(test_worker.result['loss'] <= 0.05,True)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exceptions occur.")

    def test_batch_count(self):
        # Dummy dataset
        x = np.ones((3, 1))
        y = np.zeros((3, 1))
        data = SimpleData({'training':{'inputs':x,'answers':y},
                           'validation':{'inputs':x,'answers':y}})
        # Dummy model
        inputs = Input(shape=(1,))
        outputs = Dense(1)(inputs)
        model_ = Model(inputs=inputs, outputs=outputs)
        model = SimpleModel(model_)
        compiler = SimpleCompiler('adam','mse',metrics=[StatefulMetric(BatchCount())])
        train_worker = TrainWorker(is_verbose_visible=False)
        train_wrapper = fit_wrapper_generator(batch_size=2,epochs=100,verbose=0)
        train_pipeline = Pipeline(model,data,compiler,train_worker,
                                  train_wrapper,is_prompt_visible=False)
        train_pipeline.execute()
        result = train_worker.result
        self.assertTrue(np.all(np.array(result['batch_count'])==2))
        self.assertTrue(np.all(np.array(result['val_batch_count'])==2))

    def test_sample_count(self):
        # Dummy dataset
        x = np.ones((3, 1))
        y = np.zeros((3, 1))
        data = SimpleData({'training':{'inputs':x,'answers':y},
                           'validation':{'inputs':x,'answers':y}})
        # Dummy model
        inputs = Input(shape=(1,))
        outputs = Dense(1)(inputs)
        model_ = Model(inputs=inputs, outputs=outputs)
        model = SimpleModel(model_)
        compiler = SimpleCompiler('adam','mse',metrics=[StatefulMetric(SampleCount())])
        train_worker = TrainWorker(is_verbose_visible=False)
        train_wrapper = fit_wrapper_generator(batch_size=2,epochs=100,verbose=0)
        train_pipeline = Pipeline(model,data,compiler,train_worker,
                                  train_wrapper,is_prompt_visible=False)
        train_pipeline.execute()
        result = train_worker.result
        self.assertTrue(np.all(np.array(result['sample_count'])==3))
        self.assertTrue(np.all(np.array(result['val_sample_count'])==3))