import os
import sys
from os.path import abspath, expanduser
sys.path.append(abspath(expanduser(__file__+"/../..")))
import unittest
from sequence_annotation.pipeline.processor.compiler import SimpleCompiler,AnnSeqCompiler
from sequence_annotation.pipeline.processor.model_processor import SimpleModel,ModelCreator
from sequence_annotation.pipeline.processor.data_processor import AnnSeqData,SimpleData
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from sequence_annotation.pipeline.worker.train_worker import TrainWorker
from sequence_annotation.pipeline.worker.test_worker import TestWorker
from sequence_annotation.pipeline.pipeline import Pipeline
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.data_handler.fasta import read_fasta
from sequence_annotation.data_handler.json import read_json
from sequence_annotation.data_handler.seq_converter import SeqConverter
from sequence_annotation.pipeline.wrapper import fit_generator_wrapper_generator
from sequence_annotation.pipeline.wrapper import evaluate_generator_wrapper_generator
import numpy as np
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

            pipeline.execute(1)
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
            pipeline.execute(1)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exceptions occur.")

    def test_train_test(self):
        try:
            ann_types = ["utr_5","utr_3","intron","cds","intergenic_region"]
            seq_data = np.load(abspath(expanduser(__file__+'/../data/seqs.npy'))).item()
            seqs = AnnSeqContainer().from_dict(seq_data)
            fasta = read_fasta(abspath(expanduser(__file__+'/../data/seqs.fasta')))
            model_setting_path = abspath(expanduser(__file__+'/../setting/model_setting.json'))
            model_setting = read_json(model_setting_path)
            model = ModelCreator(model_setting)
            compiler = SimpleCompiler('adam','mse')
            seq_converter = SeqConverter(codes="ATCGN", with_soft_masked_status=True)
            data = AnnSeqData({'data':{'training':{'inputs':fasta,'answers':seqs}},
                               'ANN_TYPES':ann_types},
                               seq_converter = seq_converter)
            train_worker = TrainWorker(is_verbose_visible=False)
            train_wrapper = fit_generator_wrapper_generator(batch_size=1, epochs=30,verbose=0)
            train_pipeline = Pipeline(model,data,compiler,train_worker,
                                      train_wrapper,is_prompt_visible=False)
            train_pipeline.execute(1)
            test_worker = TestWorker(is_verbose_visible=False)
            test_data = AnnSeqData({'data':{'testing':{'inputs':fasta,'answers':seqs}},
                                    'ANN_TYPES':ann_types},
                                     seq_converter = seq_converter)
            simple_model = SimpleModel(model.model)
            evaluate_wrapper = evaluate_generator_wrapper_generator(verbose=0)
            test_pipeline = Pipeline(simple_model,test_data,compiler,test_worker,
                                     evaluate_wrapper,is_prompt_visible=False)
            test_pipeline.execute(1)
            self.assertEqual(dict(test_worker.result)['loss'] <= 0.05,True)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exceptions occur.")

    def test_constant_metric(self):
        ann_types = ["utr_5","utr_3","intron","cds","intergenic_region"]
        seq_data = np.load(abspath(expanduser(__file__+'/../data/seqs.npy'))).item()
        seqs = AnnSeqContainer().from_dict(seq_data)
        fasta = read_fasta(abspath(expanduser(__file__+'/../data/seqs.fasta')))
        model_setting_path = abspath(expanduser(__file__+'/../setting/model_setting.json'))
        model_setting = read_json(model_setting_path)
        model = ModelCreator(model_setting)
        compiler = AnnSeqCompiler('adam','mse',ann_types=ann_types,metric_types=['batch_counter'])
        seq_converter = SeqConverter(codes="ATCGN", with_soft_masked_status=True)
        data = AnnSeqData({'data':{'training':{'inputs':fasta,'answers':seqs},
                                   'validation':{'inputs':fasta,'answers':seqs}},
                           'ANN_TYPES':ann_types},
                           seq_converter = seq_converter)
        train_worker = TrainWorker(is_verbose_visible=False)
        train_wrapper = fit_generator_wrapper_generator(batch_size=1,epochs=30,verbose=0)
        train_pipeline = Pipeline(model,data,compiler,train_worker,
                                  train_wrapper,is_prompt_visible=False)
        train_pipeline.execute(1)
        data = train_worker.result
        self.assertTrue(np.all(np.array(data['batch_counter_layer'])==3))
        self.assertTrue(np.all(np.array(data['val_batch_counter_layer'])==3))  
           