import os
import sys
from os.path import abspath, expanduser
import unittest
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation,Input,RNN
from keras.models import Model
from sequence_annotation.data_handler.fasta import read_fasta
from sequence_annotation.data_handler.json import read_json
from sequence_annotation.data_handler.seq_converter import SeqConverter
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.process.data_processor import AnnSeqProcessor
from sequence_annotation.process.pipeline import Pipeline
from sequence_annotation.process.data_generator import DataGenerator
from sequence_annotation.keras.function.work_generator import FitGenerator,EvaluateGenerator
from sequence_annotation.keras.process.compiler import SimpleCompiler,AnnSeqCompiler
from sequence_annotation.keras.function.stateful_metric import StatefulMetric
from sequence_annotation.keras.function.metric import BatchCount,SampleCount
from sequence_annotation.keras.process.train_worker import TrainWorker
from sequence_annotation.keras.process.test_worker import TestWorker
from sequence_annotation.keras.function.model_builder import ModelBuilder

class TestPipeline(unittest.TestCase):
    def test_simple_train(self):
        try:
            model = Sequential([
                Dense(32, input_shape=(2,)),
                Activation('relu'),
                Dense(2),
                Activation('softmax')
            ])
            compiler = SimpleCompiler('adam','binary_crossentropy')
            data = {'training':{'inputs':[[1,0]],'answers':[[1,0]]}}
            fit_generator = FitGenerator(verbose=0)
            worker = TrainWorker(fit_generator=fit_generator)
            compiler.process(model)
            pipeline = Pipeline(model,data,worker)
            pipeline.is_prompt_visible=False
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
            compiler = SimpleCompiler('adam','binary_crossentropy')
            data = {'testing':{'inputs':[[1,0]],'answers':[[1,0]]}}
            generator = EvaluateGenerator(verbose=0)
            worker = TestWorker(evaluate_generator=generator)
            compiler.process(model)
            pipeline = Pipeline(model,data,worker)
            pipeline.is_prompt_visible=False
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
            model = ModelBuilder(model_setting).build()
            train_compiler = AnnSeqCompiler('adam','mse',ann_types=ann_types,
                                            metrics=[StatefulMetric(SampleCount()),
                                                     StatefulMetric(BatchCount())])
            seq_converter = SeqConverter(codes="ATCGN", with_soft_masked_status=True)
            data = AnnSeqProcessor({'training':{'inputs':fasta,'answers':seqs}},
                               seq_converter = seq_converter).process()
            fit_generator = FitGenerator(verbose=0,epochs=30)
            train_generator = DataGenerator()
            train_generator.batch_size=1
            val_generator = DataGenerator()
            val_generator.batch_size=1
            train_worker = TrainWorker(train_generator=train_generator,
                                       val_generator=val_generator,
                                       fit_generator=fit_generator)
            train_compiler.process(model)
            train_pipeline = Pipeline(model,data,train_worker)
            train_pipeline.is_prompt_visible=False
            train_pipeline.execute()
            generator = EvaluateGenerator(verbose=0)
            test_worker = TestWorker(evaluate_generator=generator)
            test_data = AnnSeqProcessor({'testing':{'inputs':fasta,'answers':seqs}},
                                     seq_converter = seq_converter).process()
            test_pipeline = Pipeline(model,test_data,test_worker)
            test_pipeline.is_prompt_visible=False
            test_pipeline.execute()
            self.assertEqual(test_pipeline.result['loss'] <= 0.05,True)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exceptions occur.")

    def test_batch_count(self):
        # Dummy dataset
        x = np.ones((3, 1))
        y = np.zeros((3, 1))
        data = {'training':{'inputs':x,'answers':y},
                'validation':{'inputs':x,'answers':y}}
        # Dummy model
        inputs = Input(shape=(1,))
        outputs = Dense(1)(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        compiler = SimpleCompiler('adam','mse',metrics=[StatefulMetric(BatchCount())])
        fit_generator = FitGenerator(verbose=0,epochs=100)
        train_generator = DataGenerator()
        train_generator.batch_size=2
        val_generator = DataGenerator()
        val_generator.batch_size=2
        train_worker = TrainWorker(train_generator=train_generator,
                                   val_generator=val_generator,
                                   fit_generator=fit_generator)
        compiler.process(model)
        train_pipeline = Pipeline(model,data,train_worker)
        train_pipeline.is_prompt_visible=False
        train_pipeline.execute()
        result = train_pipeline.result
        self.assertTrue(np.all(np.array(result['batch_count'])==2))
        self.assertTrue(np.all(np.array(result['val_batch_count'])==2))

    def test_sample_count(self):
        # Dummy dataset
        x = np.ones((3, 1))
        y = np.zeros((3, 1))
        data = {'training':{'inputs':x,'answers':y},
                'validation':{'inputs':x,'answers':y}}
        # Dummy model
        inputs = Input(shape=(1,))
        outputs = Dense(1)(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        compiler = SimpleCompiler('adam','mse',metrics=[StatefulMetric(SampleCount())])
        fit_generator = FitGenerator(verbose=0,epochs=100)
        train_generator = DataGenerator()
        train_generator.batch_size=2
        val_generator = DataGenerator()
        val_generator.batch_size=2
        train_worker = TrainWorker(train_generator=train_generator,
                                   val_generator=val_generator,
                                   fit_generator=fit_generator)
        compiler.process(model)
        train_pipeline = Pipeline(model,data,train_worker)
        train_pipeline.is_prompt_visible=False
        train_pipeline.execute()
        result = train_pipeline.result
        self.assertTrue(np.all(np.array(result['sample_count'])==3))
        self.assertTrue(np.all(np.array(result['val_sample_count'])==3))