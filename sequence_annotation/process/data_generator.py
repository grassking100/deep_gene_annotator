from keras.utils import Sequence
import numpy as np
from abc import abstractmethod
from ..utils.utils import create_folder
from keras.preprocessing.sequence import pad_sequences
from ..genome_handler.utils import get_seq_mask

class DataGenerator(Sequence):

    def __init__(self,batch_size=None):
        self._x_data = []
        self.y_data = []
        self._indice = []
        self.extra = {}
        self.batch_size = batch_size or 32
        self.shuffle = True
        self.return_extra_info = False

    @property
    def x_data(self):
        return self._x_data

    @x_data.setter
    def x_data(self,value):
        if 'lengths' not in self.extra.keys():
            self.extra['lengths'] = [len(seq) for seq in value]
        self._x_data = value
        self._indice = np.arange(len(self._x_data))

    def __len__(self):
        length = int(np.ceil(len(self.x_data) / self.batch_size))
        return length

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._indice)

    def __getitem__(self, idx):
        indice = self._indice[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_x = np.array([self.x_data[index] for index in indice])
        batch_y = np.array([self.y_data[index] for index in indice])
        extra_info = {}
        for key,item in self.extra.items():
            ordered_item = np.array([item[index] for index in indice])
            extra_info[key] = ordered_item
        if self.return_extra_info:
            return batch_x,batch_y,extra_info
        else:
            return batch_x,batch_y

def _order(data,indice):
    return np.array([data[index] for index in indice])
        
class SeqGenerator(DataGenerator):
    """NLC means Number, Length, Channel; NCL means Number, Channel, Length"""
    def __init__(self,batch_size=None,pad_value=None,order_target=None):
        super().__init__(batch_size=None)
        self._order = 'NLC'
        self.pad_value = pad_value or {}
        self.order_target = order_target or []

    @property
    def order(self):
        return self._order
    
    @order.setter
    def order(self,value):
        if value not in ['NLC','NCL']:
            raise Exception("Invalid order:"+str(value))
        self._order = value

    def __getitem__(self, idx):
        if self.return_extra_info:
            batch_x,batch_y,extra_info = super().__getitem__(idx)
        else:
            batch_x,batch_y = super().__getitem__(idx)
        if 'inputs' in self.pad_value.keys():
            batch_x = pad_sequences(batch_x,padding='post',value=self.pad_value['inputs'])
        if 'answers' in self.pad_value.keys():
            batch_y = pad_sequences(batch_y,padding='post',value=self.pad_value['answers'])
        if self.order == 'NCL':
            if 'inputs' in self.order_target:
                batch_x = np.transpose(batch_x,[0,2,1])
            if 'answers' in self.order_target:
                batch_y = np.transpose(batch_y,[0,2,1])
        if self.return_extra_info:
            indice = self._indice[idx*self.batch_size : (idx+1)*self.batch_size]
            lengths = _order(self.extra['lengths'],indice)
            length_order = np.flip(np.argsort(lengths))
            batch_x = _order(batch_x,length_order)
            batch_y = _order(batch_y,length_order)
            new_extra_info = {}
            extra_info['mask'] = get_seq_mask(lengths)
            for key,items in extra_info.items():
                for item in items:
                    ordered_items = _order(items,length_order)
                    if key in self.pad_value.keys():
                        ordered_items = pad_sequences(ordered_items,padding='post',value=self.pad_value[key])
                    if key in self.order_target:
                        if self.order == 'NCL':
                            ordered_items = np.transpose(ordered_items,[0,2,1])
                    new_extra_info[key] = ordered_items
            return batch_x,batch_y,new_extra_info
        else:
            return batch_x,batch_y
