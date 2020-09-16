import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def order(data, indice):
    return [data[index] for index in indice]

class SeqDataset(Dataset):
    def __init__(self, data):
        self._num = None
        self._data = data
        self._keys = sorted(list(self._data.keys()))
        for key,values in data.items():
            if self._num is None:
                self._num = len(values)
            elif self._num != len(values):
                raise Exception("Wrong number between {} and {}".format(self._num,len(values)))

    @property
    def keys(self):
        return self._keys
       
    def set(self,key,value):
        if key not in self._data:
            raise
        if len(value) != len(self):
            raise
        self._data[key] = value
    
    def get(self,key):
        return self._data[key]

    def __len__(self):
        return self._num

    def __getitem__(self, index):
        returned = {}
        for key, item in self._data.items():
            returned[key] = item[index]
        return returned


class SeqCollateWrapper:
    def __init__(self,discard_ratio_min=None,discard_ratio_max=None,
                 aug_up_max=None,aug_down_max=None,concat=False,
                 shuffle=True,both_discard_order=False):
        self._discard_ratio_min = discard_ratio_min or 0
        self._discard_ratio_max = discard_ratio_max or 0
        self._aug_up_max = aug_up_max or 0
        self._aug_down_max = aug_down_max or 0
        self._native_discard_order = not both_discard_order
        self._concat = concat
        self._shuffle = shuffle
        self._discard_diff = self._discard_ratio_max - self._discard_ratio_min
        self._change = self._aug_up_max + self._aug_down_max + self._discard_ratio_min + self._discard_ratio_max 
        
        if self._discard_ratio_min < 0 or self._discard_ratio_min > 1:
            raise Exception("Invalid discard_ratio_min value")
        if self._discard_ratio_max < 0 or self._discard_ratio_max > 1:
            raise Exception("Invalid discard_ratio_max value")

    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['discard_ratio_min'] = self._discard_ratio_min
        config['discard_ratio_max'] = self._discard_ratio_max
        config['aug_up_max'] = self._aug_up_max
        config['aug_down_max'] = self._aug_down_max
        config['native_discard_order'] = self._native_discard_order
        config['concat'] = self._concat
        config['shuffle'] = self._shuffle
        return config
    
    def _truncate_data(self,data):
        returned = {'inputs':[],'answers':[],'seqs':[],
                    'lengths':[]}
        num = len(data)
        start_diff_list = np.random.randint(0, self._aug_up_max + 1, size=num)
        end_diff_list = np.random.randint(0, self._aug_down_max + 1, size=num)
        if self._native_discard_order:
            discard_order_from_starts = [True]*num
        else:
            discard_order_from_starts = np.random.randint(0, 2, size=num)==0
        discard_left_ratios = np.random.random_sample(size=num) * self._discard_diff + self._discard_ratio_min
        discard_right_ratios = np.random.random_sample(size=num) * self._discard_diff + self._discard_ratio_min
        iterator = zip(data.get('inputs'),data.get('answers'),data.get('seqs'),
                       data.get('lengths'),data.get('has_gene_statuses'))
        for index, items in enumerate(iterator):
            input_, answer, seq, length, has_gene_status = items
            if has_gene_status:
                new_start_index = start_diff_list[index]
                end_diff = end_diff_list[index]
                new_end_index = length - end_diff
            else:
                discard_left_ratio = discard_left_ratios[index]
                discard_right_ratio = discard_right_ratios[index]
                discard_order_from_start = discard_order_from_starts[index]
                if discard_order_from_start:
                    new_start_index = int(np.round(length * discard_left_ratio))
                    new_length = length - new_start_index
                    new_end_index = int(np.round(new_length *(1 - discard_right_ratio))) + new_start_index
                else:
                    new_end_index = int(np.round(length * (1-discard_right_ratio)))
                    new_length = new_end_index
                    new_start_index = int(np.round(new_length *discard_left_ratio))
            new_length = new_end_index - new_start_index
            returned['lengths'].append(new_length)
            returned['inputs'].append(input_[new_start_index:new_end_index])
            returned['answers'].append(answer[new_start_index:new_end_index])
            returned['seqs'].append(seq[new_start_index:new_end_index])
        for key,values in returned.items():
            data.set(key,values)
        return data

    def _concat_data(self,data):
        indice = list(range(len(data)))
        if len(indice)%3 == 1:
            indice += [None,None]
        elif len(indice)%3 == 2:
            indice += [None]
        if self._shuffle:
            np.random.shuffle(indice)
        step = int(len(indice)/3)
        left,middle,right = indice[:step],indice[step:step*2],indice[step*2:]
        new_data = []
        for indice in zip(left,middle,right):
            item = {}
            for index in indice:
                if index is not None:
                    if 'ids' not in item: 
                        for key in data.keys:
                            item[key] = data.get(key)[index]
                    else:
                        item['ids'] += ("_and_" + data.get('ids')[index])
                        item['inputs'] = torch.cat((item['inputs'],data.get('inputs')[index]))
                        item['answers'] = torch.cat((item['answers'],data.get('answers')[index]))
                        item['lengths'] += data.get('lengths')[index] 
                        item['seqs'] += data.get('seqs')[index]
                        item['has_gene_statuses'] = item['has_gene_statuses'] or data.get('has_gene_statuses')[index]
            new_data.append(item)
        
        returned = {}
        for key in data.keys:
            returned[key] = []
            for item in new_data:
                returned[key].append(item[key])
        return SeqDataset(returned)
    
    def _aug_data(self,data):
        if self._change > 0:
            data = self._truncate_data(data)
        if self._concat:
            data = self._concat_data(data)
        return data
    
    def __call__(self,data):
        data_ = {}
        for key in data[0].keys():
            data_[key] = []
            for item in data:
                data_[key].append(item[key])
        data = SeqDataset(data_)
        data = self._aug_data(data)
        inputs = pad_sequence(data.get('inputs'), batch_first=True)
        answers = pad_sequence(data.get('answers'), batch_first=True)
        length_order = np.flip(np.argsort(data.get('lengths')), axis=0).copy()
        data.set('ids',order(data.get('ids'), length_order))
        data.set('inputs',inputs[length_order].transpose(1, 2))
        data.set('answers',answers[length_order].transpose(1, 2))
        data.set('lengths',order(data.get('lengths'), length_order))
        data.set('seqs',order(data.get('seqs'), length_order))
        data.set('has_gene_statuses',order(data.get('has_gene_statuses'), length_order))
        return data

class SeqGenerator:
    def __init__(self, seq_collate_fn=None,*args, **kwargs):
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 1
        self.seq_collate_fn = seq_collate_fn or SeqCollateWrapper()
        self.args = args
        self.kwargs = kwargs
        self._batch_size = kwargs['batch_size']
            
    @property
    def batch_size(self):
        return self._batch_size
        
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['seq_collate_fn'] = self.seq_collate_fn.get_config()
        config['args'] = list(self.args)
        config['kwargs'] = dict(self.kwargs)
        return config

    def __call__(self, data):
        data = SeqDataset(data)
        data.set('inputs',[torch.FloatTensor(i) for i in data.get('inputs')])
        data.set('answers',[torch.LongTensor(a) for a in data.get('answers')])
        return DataLoader(data, collate_fn=self.seq_collate_fn,
                          *self.args, **self.kwargs)
