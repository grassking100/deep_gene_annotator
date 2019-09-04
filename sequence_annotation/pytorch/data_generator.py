import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from keras.preprocessing.sequence import pad_sequences

class SeqDataset(Dataset):
    def __init__(self,data=None):
        self._ids = None
        self._inputs = None
        self._answers = None
        self._lengths = None
        self._length = None
        if data is not None:
            self.data = data
        
    @property    
    def data(self):
        return self._data
    
    @data.setter    
    def data(self,data):
        self._ids = data['ids']
        self._inputs = data['inputs']
        self._answers = data['answers']
        self._lengths = data['lengths']
        self._length = len(self._ids)
        self._data = data

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        id_,input_ = self._ids[idx],self._inputs[idx]
        answer_,length = self._answers[idx],self._lengths[idx]
        return id_, input_,answer_, length

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
    
def seq_collate_wrapper(augmentation_max=None):
    augmentation_max = augmentation_max or 0
    def seq_collate_fn(data):
        transposed_data  = list(zip(*data))
        ids, inputs, answers, lengths = transposed_data
        if augmentation_max > 0:
            inputs,answers,lengths = augmentation_seqs(inputs,answers,lengths,augmentation_max)
        inputs = pad_sequences(inputs,padding='post')
        answers = pad_sequences(answers,padding='post')
        length_order = np.flip(np.argsort(lengths))
        ids = order(ids,length_order)
        inputs = order(inputs,length_order)
        answers = order(answers,length_order)
        lengths = order(lengths,length_order)
        inputs = torch.FloatTensor(inputs).transpose(1,2)
        answers = torch.LongTensor(answers).transpose(1,2)
        return ids, inputs, answers, lengths
    return seq_collate_fn
    
class SeqLoader(DataLoader):
    def __init__(self,dataset,augmentation_max=None,*args,**kwargs):
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = seq_collate_wrapper(augmentation_max)
        if 'shuffle' not in kwargs:
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
    