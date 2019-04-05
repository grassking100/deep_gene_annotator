import unittest
from sequence_annotation.keras.process.compiler import SimpleCompiler,AnnSeqCompiler
from keras.models import Sequential
from keras.layers import Dense, Activation,GRU,Convolution1D

class TestPipeline(unittest.TestCase):
    def test_simple_compiler(self):
        try:
            model = Sequential([
                Dense(32, input_shape=(2,)),
                Activation('relu'),
                Dense(2),
                Activation('softmax')
            ])
            compiler = SimpleCompiler('adam','binary_crossentropy')
            compiler.before_process()
            compiler.process(model)
            compiler.after_process()
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
    def test_ann_seq_compiler(self):
        try:
            model = Sequential([
                GRU(2, input_shape=(None,2),return_sequences=True),
                Convolution1D(2,1,activation='softmax')
            ])
            compiler = AnnSeqCompiler('adam','categorical_crossentropy',
                                      values_to_ignore=-1,ann_types=['T','N'],
                                      dynamic_weight_method='reversed_count_weight')
            compiler.before_process()
            compiler.process(model)
            compiler.after_process()
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")