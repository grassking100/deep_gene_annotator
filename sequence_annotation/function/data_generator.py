from keras.utils import Sequence
import numpy as np
from abc import abstractmethod
from ..utils.utils import padding
from ..utils.utils import create_folder,padding

class Generator(Sequence):

    def __init__(self, x_set, y_set, batch_size,epoch_shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = epoch_shuffle
        self.indices = np.arange(len(self.x))
        
    def __len__(self):
        length = int(np.ceil(len(self.x) / self.batch_size))
        return length

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    
class DataGenerator(Generator):

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_x = np.array([self.x[index] for index in indices])
        batch_y = np.array([self.y[index] for index in indices])
        return batch_x,batch_y

class SeqGenerator(Generator):

    def __init__(self, x_set, y_set,lengths,batch_size,epoch_shuffle=True,
                 return_len=False,order=None,padding_value=None):
        super().__init__(x_set=x_set, y_set=y_set,
                         batch_size=batch_size,
                         epoch_shuffle=epoch_shuffle)
        self._order = order or 'NLC'
        self._padding_value = padding_value
        # NLC means Number,Length,Channel
        # NCL means Number,Channel,Length
        if self._order not in ['NLC','NCL']:
            raise Exception("Invalid order:"+str(self._order))
        self._return_len = return_len
        self._lengths = lengths

    def __getitem__(self, idx):
        indice = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        length_order = np.flip(np.argsort([self._lengths[index] for index in indice]))
        indice = [indice[index] for index in length_order]
        batch_x = [self.x[index] for index in indice]
        batch_y = [self.y[index] for index in indice]
        batch_length = np.array([self._lengths[index] for index in indice])
        if self._padding_value is not None:
            batch_x, batch_y = padding(batch_x, batch_y, self._padding_value)
        if self._order == 'NCL':
            batch_x = np.transpose(batch_x,[0,2,1])
            if len(np.array(batch_y).shape)==3:
                batch_y = np.transpose(batch_y,[0,2,1])
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        if self._return_len:
            return batch_x,batch_y,batch_length
        else:
            return batch_x,batch_y                


def seq_generator_wrapper(*args,**kwargs):
    def seq_generator(inputs,answers,lengths):
        return SeqGenerator(inputs,answers,lengths,*args,**kwargs)
    return seq_generator