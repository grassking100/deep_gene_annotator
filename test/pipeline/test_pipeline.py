import os
import sys
from os.path import abspath, expanduser
sys.path.append(abspath(expanduser(__file__+"/../..")))
import unittest
from sequence_annotation.pipeline.processor.compiler import SimpleCompiler
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
import numpy as np
class TestPipeline(unittest.TestCase):
    def test_simple_train(self):
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
        pipeline = Pipeline(simple_model,data,compiler,worker,is_prompt_visible=False)
        pipeline.execute(1)
    def test_simple_test(self):
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
        pipeline = Pipeline(simple_model,data,compiler,worker,is_prompt_visible=False)
        pipeline.execute(1)
    def test_train_test(self):
        try:
            seq_data = np.load(abspath(expanduser(__file__+'/../data/seqs.npy'))).item()
            seqs = AnnSeqContainer().from_dict(seq_data)
            fasta = read_fasta(abspath(expanduser(__file__+'/../data/seqs.fasta')))
            model_setting_path = abspath(expanduser(__file__+'/../setting/model_setting.json'))
            model_setting = read_json(model_setting_path)
            model = ModelCreator(model_setting)
            compiler = SimpleCompiler('adam','mse')
            seq_converter = SeqConverter(codes="ATCGN", with_soft_masked_status=True)
            data = AnnSeqData({'data':{'training':{'inputs':fasta,'answers':seqs}},
                               'ANN_TYPES':["utr_5","utr_3","intron","cds","intergenic_region"]},
                               seq_converter = seq_converter)
            train_worker = TrainWorker(is_verbose_visible=False,batch_size=1, epoch=30,
                                       validation_split=0.0, use_generator=True)
            train_pipeline = Pipeline(model,data,compiler,train_worker,is_prompt_visible=False)
            train_pipeline.execute(1)
            test_worker = TestWorker(is_verbose_visible=False)
            test_data = AnnSeqData({'data':{'testing':{'inputs':fasta,'answers':seqs}},
                                    'ANN_TYPES':["utr_5","utr_3","intron","cds","intergenic_region"]},
                                     seq_converter = seq_converter)
            simple_model = SimpleModel(model.model)
            test_pipeline = Pipeline(simple_model,test_data,compiler,test_worker,is_prompt_visible=False)
            test_pipeline.execute(1)
            self.assertEqual(dict(test_worker.result)['loss'] <= 0.05,True)
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")