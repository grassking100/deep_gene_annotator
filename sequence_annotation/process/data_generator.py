import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class SeqDataset(Dataset):
    def __init__(self, data):
        self._ids = data['ids']
        self._num = len(self._ids)
        self._inputs = data['inputs']
        self._has_gene_statuses = None
        self._answers = None
        self._seqs = None
        self._lengths = data['lengths']
        if 'seqs' in data:
            self._seqs = data['seqs']
        if 'answers' in data:
            self._answers = data['answers']
            self._has_gene_statuses = data['has_gene_statuses']

    @property
    def data(self):
        data = {}
        keys = [
            'ids', 'inputs', 'answers', 'lengths',
            'seqs','has_gene_statuses'
        ]
        for key in keys:
            data[key] = getattr(self, "_" + key)
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
    def has_gene_statuses(self):
        return self._has_gene_statuses

    def __len__(self):
        return self._num

    def __getitem__(self, index):
        returned = {}
        for key, list_ in self.data.items():
            if list_ is not None:
                returned[key] = list_[index]
            else:
                returned[key] = None
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


def concat_seq(ids,inputs,answers,lengths,seqs,has_gene_statuses):
    indice = list(range(len(ids)))
    if len(indice)%3 == 1:
        indice += [None,None]
    elif len(indice)%3 == 2:
        indice += [None]
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
    answers = [item['answers'] for item in data]
    lengths = [item['lengths'] for item in data]
    seqs = [item['seqs'] for item in data]
    has_gene_statuses = [item['has_gene_statuses'] for item in data]
    return ids,inputs,answers,lengths,seqs,has_gene_statuses

def seq_collate_wrapper(discard_ratio_min=None,
                        discard_ratio_max=None,
                        augment_up_max=None,
                        augment_down_max=None,
                        concat=False):
    discard_ratio_min = discard_ratio_min or 0
    discard_ratio_max = discard_ratio_max or 0
    augment_up_max = augment_up_max or 0
    augment_down_max = augment_down_max or 0

    if discard_ratio_min < 0 or discard_ratio_min > 1:
        raise Exception("Invalid discard_ratio_min value")
    if discard_ratio_max < 0 or discard_ratio_max > 1:
        raise Exception("Invalid discard_ratio_max value")

    def seq_collate_fn(data):
        answers = reordered_answers = None
        data = _get_list(data)
        ids,inputs,answers,lengths,seqs,has_gene_statuses = data
        change = augment_up_max + augment_down_max + discard_ratio_min + discard_ratio_max 
        if change > 0:
            result = augment_seqs(inputs, answers, seqs, lengths, has_gene_statuses,
                                  discard_ratio_min, discard_ratio_max,
                                  augment_up_max, augment_down_max)
            inputs, answers,seqs, lengths = result

        if concat:
            concat_result = concat_seq(ids,inputs,answers,lengths,seqs,has_gene_statuses)
            ids,inputs,answers,lengths,seqs,has_gene_statuses = _get_list(concat_result)
            
        inputs = pad_sequence(inputs, batch_first=True)
        if answers[0] is not None:
            answers = pad_sequence(answers, batch_first=True)
        length_order = np.flip(np.argsort(lengths), axis=0).copy()
        reordered_ids = order(ids, length_order)
        reordered_inputs = inputs[length_order].transpose(1, 2)
        
        if answers[0] is not None:
            reordered_answers = answers[length_order].transpose(1, 2)
        reordered_lengths = order(lengths, length_order)
        reordered_seqs = order(seqs, length_order)
        reordered_has_gene_statuses = order(has_gene_statuses, length_order)
        data = {
            'ids': reordered_ids,
            'inputs': reordered_inputs,
            'answers': reordered_answers,
            'lengths': reordered_lengths,
            'seqs': reordered_seqs,
            'has_gene_statuses': reordered_has_gene_statuses
        }
        return SeqDataset(data)

    return seq_collate_fn


class SeqGenerator:
    def __init__(self, seq_collate_fn=None,*args, **kwargs):
        self.seq_collate_fn = seq_collate_fn or seq_collate_wrapper()
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dataset):
        dataset = SeqDataset(dataset)
        dataset._inputs = [torch.FloatTensor(i) for i in dataset.inputs]
        if dataset.answers is not None:
            dataset._answers = [torch.LongTensor(a) for a in dataset.answers]
        return DataLoader(dataset, collate_fn=self.seq_collate_fn,
                          *self.args, **self.kwargs)

