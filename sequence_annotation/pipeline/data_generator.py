from keras.utils import Sequence
import numpy as np
import math
from ..utils.utils import padding

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size,padding=None,epoch_shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.padding = padding
        self.shuffle = epoch_shuffle
        self.indices = np.arange(len(self.x))
    def __len__(self):
        length = int(np.ceil(len(self.x) / self.batch_size))
        return length
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_x = np.array([self.x[index] for index in indices])
        batch_y = np.array([self.y[index] for index in indices])
        if self.padding is not None:
            return padding(batch_x,batch_y,self.padding)
        else:
            return batch_x,batch_y
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)