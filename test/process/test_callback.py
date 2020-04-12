import unittest
from sequence_annotation.process.callback import DataCallback, MeanRecorder, DataHolder


class TestCallback(unittest.TestCase):
    def test_data_callback(self):
        class BatchCounter(DataCallback):
            def __init__(self, prefix=None):
                super().__init__(prefix)
                self.counter = 0

            @property
            def data(self):
                return self.counter

            def _reset(self):
                self.counter = 0

            def on_batch_begin(self, **kwargs):
                super().on_batch_begin(**kwargs)
                self.counter += 1

        callback = BatchCounter()
        callback.counter = 10
        self.assertEqual(10, callback.data)
        callback.on_work_begin()
        self.assertEqual(0, callback.data)
        callback.counter = 10
        self.assertEqual(10, callback.data)
        callback.on_epoch_begin(1)
        self.assertEqual(10, callback.data)
        callback.on_batch_begin()
        self.assertEqual(11, callback.data)

    def test_mean_recorder(self):
        callback = MeanRecorder()
        callback.on_epoch_begin()
        callback.on_batch_end({'loss': 10})
        self.assertEqual(10, callback.data['loss'])
        callback.on_batch_end({'loss': 20})
        self.assertEqual(15, callback.data['loss'])
        callback.on_epoch_begin()
        callback.on_batch_end({'loss': 20})
        self.assertEqual(20, callback.data['loss'])

    def test_data_holder(self):
        callback = DataHolder()
        callback.on_epoch_begin()
        callback.on_batch_end({'loss': 10})
        self.assertEqual(10, callback.data['loss'])
        callback.on_batch_end({'loss': 20})
        self.assertEqual(20, callback.data['loss'])
        callback.on_epoch_begin()
        self.assertFalse('loss' in callback.data)
