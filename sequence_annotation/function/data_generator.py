from keras.utils import Sequence
import numpy as np
from abc import abstractmethod
from ..utils.utils import create_folder,padding
from keras.preprocessing.sequence import pad_sequences

class DataGenerator(Sequence):

    def __init__(self, x_set, y_set,extra=None,batch_size=32, epoch_shuffle=True,
                 return_extra_info=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = epoch_shuffle
        self.indice = np.arange(len(self.x))
        self._return_extra_info = return_extra_info
        self._extra = extra or {}

    def __len__(self):
        length = int(np.ceil(len(self.x) / self.batch_size))
        return length

    def on_epoch_end(self):
        if self.shuffle:
            print("Shuffle data")
            np.random.shuffle(self.indice)

    def __getitem__(self, idx):
        indice = self.indice[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_x = np.array([self.x[index] for index in indice])
        batch_y = np.array([self.y[index] for index in indice])
        extra_info = {}
        for key,item in self._extra.items():
            ordered_item = np.array([item[index] for index in indice])
            extra_info[key] = ordered_item
        if self._return_extra_info:
            return batch_x,batch_y,extra_info
        else:
            return batch_x,batch_y

class SeqGenerator(DataGenerator):
    """NLC means Number, Length, Channel; NCL means Number, Channel, Length"""
    def __init__(self,x_set, y_set, extra=None, batch_size=32, epoch_shuffle=True,
                 return_extra_info=False,order=None,order_target=None,pad_value=None):
        if 'lengths' not in extra.keys():
            extra['lengths'] = lengths or [len(seq) for seq in x_set]
        super().__init__(x_set=x_set, y_set=y_set,
                         extra=extra,
                         batch_size=batch_size,
                         epoch_shuffle=epoch_shuffle,
                         return_extra_info=return_extra_info)
        self._order = order or 'NLC'
        self._pad_value = pad_value or {}
        self._order_target = order_target or []
        if self._order not in ['NLC','NCL']:
            raise Exception("Invalid order:"+str(self._order))

    def __getitem__(self, idx):
        if self._return_extra_info:
            batch_x,batch_y,extra_info = super().__getitem__(idx)
        else:
            batch_x,batch_y = super().__getitem__(idx)
        if 'inputs' in self._pad_value.keys():
            batch_x = pad_sequences(batch_x,padding='post',
                                    value=self._pad_value['inputs'])
        if 'answers' in self._pad_value.keys():
            batch_y = pad_sequences(batch_y,padding='post',
                                    value=self._pad_value['answers'])
        if self._order == 'NCL':
            if 'inputs' in self._order_target:
                batch_x = np.transpose(batch_x,[0,2,1])
            if 'answers' in self._order_target:
                batch_y = np.transpose(batch_y,[0,2,1])
        if self._return_extra_info:
            indice = self.indice[idx*self.batch_size : (idx+1)*self.batch_size]
            item_lengths = [self._extra['lengths'][index] for index in indice]
            item_length_order = np.flip(np.argsort(item_lengths))
            batch_x = np.array([batch_x[index] for index in item_length_order])
            batch_y = np.array([batch_y[index] for index in item_length_order])
            new_extra_info = {}
            for key,items in extra_info.items():
                for item in items:
                    ordered_items = np.array([items[index] for index in item_length_order])
                    if key in self._pad_value.keys():
                        ordered_items = pad_sequences(ordered_items,padding='post',
                                                      value=self._pad_value[key])
                    if key in self._order_target:
                        if self._order == 'NCL':
                            ordered_items = np.transpose(ordered_items,[0,2,1])
                    new_extra_info[key] = ordered_items
            return batch_x,batch_y,new_extra_info
        else:
            return batch_x,batch_y
