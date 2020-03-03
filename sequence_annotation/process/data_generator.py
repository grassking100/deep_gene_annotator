import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class SeqDataset(Dataset):
    def __init__(self,data):
        self._ids = data['ids']
        self._inputs = data['inputs']
        self._answers = data['answers']
        self._lengths = data['lengths']
        self._seqs = data['seqs']  
        self._strands = data['strands']  
        self._has_gene_statuses = data['has_gene_statuses']
        self._num = len(self._ids)

    @property    
    def data(self):
        data = {}
        keys = ['ids','inputs','answers','lengths','seqs','strands','has_gene_statuses']
        for key in keys:
            data[key] = getattr(self,"_"+key)
        return data

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
    def strands(self):
        return self._strands

    @property    
    def has_gene_statuses(self):
        return self._has_gene_statuses

    def __len__(self):
        return self._num

    def __getitem__(self, index):
        returned = {}
        for key,list_ in self.data.items():
            returned[key] = list_[index]
        return returned

def augment_seqs(inputs,answers,lengths,has_gene_statuses,
                 discard_ratio_min,discard_ratio_max,
                 augment_up_max,augment_down_max):
    inputs_ = []
    answers_ = []
    num = len(lengths)
    start_diff_list = np.random.randint(0,augment_up_max+1,size=num)
    end_diff_list = np.random.randint(0,augment_down_max+1,size=num)
    discard_diff = discard_ratio_max-discard_ratio_min
    discard_left_ratio_list = np.random.random_sample(size=num)*discard_diff+discard_ratio_min
    discard_right_ratio_list = np.random.random_sample(size=num)*discard_diff+discard_ratio_min
    new_lengths = []
    for index,(input_,answer,length,has_gene_status) in enumerate(zip(inputs,answers,lengths,has_gene_statuses)):
        if has_gene_status:
            new_start_index = start_diff_list[index]
            end_diff = end_diff_list[index]
            new_end_index = length - end_diff
        else:
            discard_left_ratio = discard_left_ratio_list[index]
            discard_right_ratio = discard_right_ratio_list[index]
            new_start_index = int(np.round(length*discard_left_ratio))
            new_length = length - new_start_index
            new_end_index = int(np.round(new_length*(1-discard_right_ratio))) + new_start_index
        input_ = input_[new_start_index:new_end_index]
        answer = answer[new_start_index:new_end_index]
        length = input_.shape[0]
        inputs_.append(input_)
        answers_.append(answer)
        new_lengths.append(length)
    return inputs_,answers_,new_lengths
    
def order(data,indice):
    return [data[index] for index in indice]
    
def seq_collate_wrapper(discard_ratio_min=None,discard_ratio_max=None,augment_up_max=None,augment_down_max=None):
    discard_ratio_min = discard_ratio_min or 0
    discard_ratio_max = discard_ratio_max or 0
    augment_up_max = augment_up_max or 0
    augment_down_max = augment_down_max or 0
    
    if discard_ratio_min < 0 or discard_ratio_min > 1:
        raise Exception("Invalid discard_ratio_min value")
    if discard_ratio_max < 0 or discard_ratio_max > 1:
        raise Exception("Invalid discard_ratio_max value")

    def seq_collate_fn(data):
        ids = [item['ids'] for item in data]
        inputs = [item['inputs'] for item in data]
        answers = [item['answers'] for item in data]
        lengths = [item['lengths'] for item in data]
        seqs = [item['seqs'] for item in data]
        strands = [item['strands'] for item in data]
        has_gene_statuses = [item['has_gene_statuses'] for item in data]
        if (augment_up_max+augment_down_max+discard_ratio_min+discard_ratio_max) > 0:
            result = augment_seqs(inputs,answers,lengths,has_gene_statuses,
                                  discard_ratio_min,discard_ratio_max,
                                  augment_up_max,augment_down_max)
            inputs,answers,lengths = result
        inputs = pad_sequence(inputs,batch_first=True)
        answers = pad_sequence(answers,batch_first=True)
        length_order = np.flip(np.argsort(lengths),axis=0).copy()
        reordered_ids = order(ids,length_order)
        reordered_inputs = inputs[length_order].transpose(1,2)
        reordered_answers = answers[length_order].transpose(1,2)
        reordered_lengths = order(lengths,length_order)
        reordered_seqs = order(seqs,length_order)
        reordered_strands = order(strands,length_order)
        reordered_has_gene_statuses = order(has_gene_statuses,length_order)
        data = {'ids':reordered_ids,
                'inputs':reordered_inputs,
                'answers':reordered_answers,
                'lengths':reordered_lengths,
                'seqs':reordered_seqs,
                'strands':reordered_strands,
                'has_gene_statuses':reordered_has_gene_statuses
        }
        return data
    return seq_collate_fn

class SeqGenerator:
    def __init__(self,*args,**kwargs):
        self.args=args
        self.kwargs=kwargs

    def __call__(self,dataset):
        dataset._inputs = [torch.FloatTensor(i) for i in dataset.inputs]
        dataset._answers = [torch.LongTensor(a) for a in dataset.answers]
        return DataLoader(dataset,*self.args,**self.kwargs) 
