import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class SeqDataset(Dataset):
    def __init__(self, data):
        self._ids = None
        self._num = None
        self._has_gene_statuses = None
        self._answers = None
        self._seqs = None
        self._signals = None
        self._inputs = None
        self._lengths = None

        if 'ids' in data:
            self._ids = data['ids']
            self._num = len(self._ids)

        if 'inputs' in data:
            self._inputs = data['inputs']
            self._lengths = data['lengths']
            self._num = len(self._inputs)
            
        if 'signals' in data:
            self._signals = {}
            self._num = 0
            for key,seqs in data['signals'].items():
                self._signals[key] = seqs['inputs']
                self._num = max(self._num,len(seqs['inputs']))
            
        if 'seqs' in data:
            self._seqs = data['seqs']
            self._num = len(self._seqs)
            
        if 'answers' in data:
            self._answers = data['answers']
            self._has_gene_statuses = data['has_gene_statuses']
            self._num = len(self._answers)
            
    @property
    def data(self):
        data = {}
        keys = [
            'ids', 'inputs', 'answers', 'lengths',
            'seqs','has_gene_statuses', 'signals'
        ]
        for key in keys:
            data[key] = getattr(self, "_" + key)
        return data

    @property
    def signals(self):
        return self._signals
    
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
    def has_gene_statuses(self):
        return self._has_gene_statuses

    def __len__(self):
        return self._num

    def __getitem__(self, index):
        returned = {}
        for key, item in self.data.items():
            if item is not None:
                if isinstance(item,dict):
                    returned[key] = {}
                    for item_id,elements in item.items():
                        returned[key][item_id] = elements[index%len(elements)]
                else:
                    returned[key] = item[index]
        return returned
    
def augment_seqs(inputs, answers,seqs, lengths, has_gene_statuses,
                 discard_ratio_min, discard_ratio_max, augment_up_max,
                 augment_down_max):
    inputs_ = []
    answers_ = []
    seqs_ = []
    new_lengths = []
    num = len(lengths)
    start_diff_list = np.random.randint(0, augment_up_max + 1, size=num)
    end_diff_list = np.random.randint(0, augment_down_max + 1, size=num)
    discard_diff = discard_ratio_max - discard_ratio_min
    discard_left_ratio_list = np.random.random_sample(
        size=num) * discard_diff + discard_ratio_min
    discard_right_ratio_list = np.random.random_sample(
        size=num) * discard_diff + discard_ratio_min
    for index, (input_, answer,seq, length, has_gene_status) in enumerate(
            zip(inputs, answers,seqs, lengths, has_gene_statuses)):
        if has_gene_status:
            new_start_index = start_diff_list[index]
            end_diff = end_diff_list[index]
            new_end_index = length - end_diff
        else:
            discard_left_ratio = discard_left_ratio_list[index]
            discard_right_ratio = discard_right_ratio_list[index]
            new_start_index = int(np.round(length * discard_left_ratio))
            new_length = length - new_start_index
            new_end_index = int(
                np.round(new_length *
                         (1 - discard_right_ratio))) + new_start_index
        input_ = input_[new_start_index:new_end_index]
        answer = answer[new_start_index:new_end_index]
        seq = seq[new_start_index:new_end_index]
        length = input_.shape[0]
        inputs_.append(input_)
        answers_.append(answer)
        seqs_.append(seq)
        new_lengths.append(length)
    return inputs_, answers_,seqs_, new_lengths


def order(data, indice):
    return [data[index] for index in indice]


def concat_seq(ids,inputs,answers,lengths,seqs,has_gene_statuses,shuffle=True):
    indice = list(range(len(ids)))
    if len(indice)%3 == 1:
        indice += [None,None]
    elif len(indice)%3 == 2:
        indice += [None]
    if shuffle:
        np.random.shuffle(indice)
    step = int(len(indice)/3)
    left = indice[:step]
    middle = indice[step:step*2]
    right = indice[step*2:]
    returned = []
    for indice in zip(left,middle,right):
        item = {}
        for index in indice:
            if index is not None:
                if 'ids' not in item:
                    item['ids'] = ids[index]
                    item['inputs'] = inputs[index]
                    item['answers'] = answers[index]
                    item['lengths'] = lengths[index]
                    item['seqs'] = seqs[index]
                    item['has_gene_statuses'] = has_gene_statuses[index]
                else:
                    item['ids'] += ("_and_" + ids[index])
                    item['inputs'] = torch.cat((item['inputs'],inputs[index]))
                    item['answers'] = torch.cat((item['answers'],answers[index]))
                    item['lengths'] += lengths[index] 
                    item['seqs'] += seqs[index]
                    item['has_gene_statuses'] = item['has_gene_statuses'] or has_gene_statuses[index]
        returned.append(item)
    return returned

def _get_list(data):
    ids = [item['ids'] for item in data]
    inputs = [item['inputs'] for item in data]
    if 'answers' in data[0]:
        answers = [item['answers'] for item in data]
        has_gene_statuses = [item['has_gene_statuses'] for item in data]
    else:
        answers = None
        has_gene_statuses = None
        
    lengths = [item['lengths'] for item in data]
    seqs = [item['seqs'] for item in data]
   
    return ids,inputs,answers,lengths,seqs,has_gene_statuses

def aug_seq(data,discard_ratio_min=None,discard_ratio_max=None,
            augment_up_max=None,augment_down_max=None,
            concat=False,shuffle=True):
    ids,inputs,answers,lengths,seqs,has_gene_statuses = data
    change = augment_up_max + augment_down_max + discard_ratio_min + discard_ratio_max 
    if change > 0:
        result = augment_seqs(inputs, answers, seqs, lengths, has_gene_statuses,
                              discard_ratio_min, discard_ratio_max,
                              augment_up_max, augment_down_max)
        inputs, answers,seqs, lengths = result
    if concat:
        concat_result = concat_seq(ids,inputs,answers,lengths,seqs,has_gene_statuses,shuffle)
        ids,inputs,answers,lengths,seqs,has_gene_statuses = _get_list(concat_result)
    return ids,inputs,answers,lengths,seqs,has_gene_statuses

class SeqCollateWrapper:
    def __init__(self,discard_ratio_min=None,discard_ratio_max=None,
                 augment_up_max=None,augment_down_max=None,
                 concat=False,shuffle=True):
        
        self.discard_ratio_min = discard_ratio_min or 0
        self.discard_ratio_max = discard_ratio_max or 0
        self.augment_up_max = augment_up_max or 0
        self.augment_down_max = augment_down_max or 0
        self.concat = concat
        self.shuffle = shuffle
        
        if self.discard_ratio_min < 0 or self.discard_ratio_min > 1:
            raise Exception("Invalid discard_ratio_min value")
        if self.discard_ratio_max < 0 or self.discard_ratio_max > 1:
            raise Exception("Invalid discard_ratio_max value")

    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['discard_ratio_min'] = self.discard_ratio_min
        config['discard_ratio_max'] = self.discard_ratio_max
        config['augment_up_max'] = self.augment_up_max
        config['augment_down_max'] = self.augment_down_max
        config['concat'] = self.concat
        config['shuffle'] = self.shuffle
        return config


    def __call__(self,data):
        reordered_has_gene_statuses = answers = reordered_answers = None
        data = _get_list(data)
        data = aug_seq(data,discard_ratio_min=self.discard_ratio_min,
                       discard_ratio_max=self.discard_ratio_max,
                       augment_up_max=self.augment_up_max,
                       augment_down_max=self.augment_down_max,
                       concat=self.concat,shuffle=self.shuffle
                      )
        ids,inputs,answers,lengths,seqs,has_gene_statuses = data
        inputs = pad_sequence(inputs, batch_first=True)
        length_order = np.flip(np.argsort(lengths), axis=0).copy()
        reordered_ids = order(ids, length_order)
        reordered_inputs = inputs[length_order].transpose(1, 2)
        reordered_lengths = order(lengths, length_order)
        reordered_seqs = order(seqs, length_order)
        returned = {
            'ids': reordered_ids,
            'inputs': reordered_inputs,
            'lengths': reordered_lengths,
            'seqs': reordered_seqs
        }        
        
        if answers is not None:
            answers = pad_sequence(answers, batch_first=True)
            returned['answers'] = answers[length_order].transpose(1, 2)
            returned['has_gene_statuses'] = order(has_gene_statuses, length_order)

        return SeqDataset(returned)

class SeqGenerator:
    def __init__(self, seq_collate_fn=None,*args, **kwargs):
        self.seq_collate_fn = seq_collate_fn or SeqCollateWrapper()
        self.args = args
        self.kwargs = kwargs
        
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['seq_collate_fn'] = self.seq_collate_fn.get_config()
        config['args'] = list(self.args)
        config['kwargs'] = dict(self.kwargs)
        return config

    def __call__(self, dataset):
        dataset = SeqDataset(dataset)
        dataset._inputs = [torch.FloatTensor(i) for i in dataset.inputs]
        if dataset.answers is not None:
            dataset._answers = [torch.LongTensor(a) for a in dataset.answers]
        return DataLoader(dataset, collate_fn=self.seq_collate_fn,
                          *self.args, **self.kwargs)

