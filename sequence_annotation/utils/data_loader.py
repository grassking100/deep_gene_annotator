"""This submodule provide abstract class which load data to container"""
from abc import ABCMeta
from keras.preprocessing.sequence import pad_sequences

class DataLoader(metaclass=ABCMeta):
    """A Mediator make container have their data"""
    def _alignment(self, input_sequences, output_sequences, padding_signal):
        align_x = pad_sequences(input_sequences, maxlen=None, padding='post',
                                value=0)
        align_y = pad_sequences(output_sequences, maxlen=None, padding='post',
                                value=padding_signal)
        return (align_x, align_y)

class TrainDataLoader(DataLoader):
    """A Mediator make trainer have their training and validation data"""
    def load(self, container, train_seq, train_ann, val_seq, val_ann, padding_signal):
        """Load and padding data into same length to container"""
        (train_x, train_y) = self._alignment(train_seq, train_ann, padding_signal)
        (validation_x, validation_y) = self._alignment(val_seq, val_ann, padding_signal)
        container.load_data(train_x, 'train_x')
        container.load_data(train_y, 'train_y')
        container.load_data(validation_x, 'validation_x')
        container.load_data(validation_y, 'validation_y')

class TestDataLoader(DataLoader):
    """A Mediator make tester have their testing data"""
    def load(self, container, seq, ann, padding_signal):
        """Load and padding data into same length to container"""
        (input_sequences, output_sequences) = self._alignment(seq, ann, padding_signal)
        container.load_data(input_sequences, 'x')
        container.load_data(output_sequences, 'y')
