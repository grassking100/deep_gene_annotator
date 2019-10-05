import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from keras.preprocessing.sequence import pad_sequences

class SeqDataset(Dataset):
    def __init__(self,data=None):
        self._ids = None
        self._inputs = None
        self._answers = None
        self._lengths = None
        self._length = None
        self._seqs = None
        if data is not None:
            self.data = data
 
    @property    
    def ids(self):
        return self._ids
    
    @property    
    def inputs(self):
        return self._inputs
    
    @property    
    def answers(self):
        return self._answers
    
    @property    
    def lengths(self):
        return self._lengths
    
    @property    
    def seqs(self):
        return self._seqs
        
    @property    
    def data(self):
        return self._data
    
    @data.setter    
    def data(self,data):
        if isinstance(data,dict):
            self._ids = data['ids']
            self._inputs = data['inputs']
            self._answers = data['answers']
            self._lengths = data['lengths']
            self._seqs = data['seqs']  
        elif isinstance(data,list) or isinstance(data,tuple):
            self._ids = data[0]
            self._inputs = data[1]
            self._answers = data[2]
            self._lengths = data[3]
            self._seqs = data[4]
        else:
            raise Exception("Data should be dictionary or list")
        self._length = len(self._ids)
        self._data = data

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        id_,input_ = self._ids[idx],self._inputs[idx]
        answer_,length = self._answers[idx],self._lengths[idx]
        seqs_ = self._seqs[idx]
        return id_, input_,answer_, length, seqs_

def augmentation_seqs(inputs,answers,lengths,augmentation_max):
    inputs_ = []
    answers_ = []
    lengths_ = []
    for input_,answer,length in zip(inputs,answers,lengths):
        start_diff = np.random.randint(0,augmentation_max+1)
        end_diff = np.random.randint(0,augmentation_max+1)
        length = length - start_diff - end_diff
        if length <= 0:
            raise Exception("Got non-positive length")
        input_ = input_[start_diff:length+start_diff]
        answer = answer[start_diff:length+start_diff]
        inputs_.append(input_)
        answers_.append(answer)
        lengths_.append(length)
    return inputs_,answers_,lengths_
    
def order(data,indice):
    return np.array([data[index] for index in indice])
    
def seq_collate_wrapper(augmentation_max=None,padding_first=False):
    augmentation_max = augmentation_max or 0
    def seq_collate_fn(data):
        transposed_data  = list(zip(*data))
        ids, inputs, answers, lengths, seqs = transposed_data
        if not padding_first:
            if augmentation_max > 0:
                inputs,answers,lengths = augmentation_seqs(inputs,answers,lengths,augmentation_max)
            inputs = pad_sequences(inputs,padding='post')
            answers = pad_sequences(answers,padding='post')
        length_order = np.flip(np.argsort(lengths))
        ids = order(ids,length_order)
        inputs = order(inputs,length_order)
        answers = order(answers,length_order)
        lengths = order(lengths,length_order)
        seqs = order(seqs,length_order)
        inputs = torch.FloatTensor(inputs).transpose(1,2)
        answers = torch.LongTensor(answers).transpose(1,2)
        return ids, inputs, answers, lengths, seqs
    return seq_collate_fn
    
class SeqLoader(DataLoader):
    def __init__(self,dataset,augmentation_max=None,padding_first=False,*args,**kwargs):
        if padding_first:
            if augmentation_max is None or augmentation_max == 0:
                dataset._inputs = pad_sequences(dataset._inputs,padding='post')
                dataset._answers = pad_sequences(dataset._answers,padding='post')
            else:
                raise Exception("The augmentation_max should be zero or None when the padding_first is True")
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = seq_collate_wrapper(augmentation_max,padding_first)
        if 'shuffle' not in kwargs:
            warnings.warn("Set the shuffle to True by default")
            kwargs['shuffle'] = True

        super().__init__(dataset,*args,**kwargs)

        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:    
            batch_size = 1
        
        self._length = int(np.ceil(len(dataset) /batch_size))

    def __len__(self):
        return self._length

class SeqGenerator:
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs

    def __call__(self,dataset):
        return SeqLoader(dataset,*self.args,**self.kwargs) 
    