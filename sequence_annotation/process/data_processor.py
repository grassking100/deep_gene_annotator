import torch
import numpy as np
from ..genome_handler.ann_genome_processor import genome2dict_vec
from ..utils.seq_converter import SeqConverter
from ..utils.exception import LengthNotEqualException


class AnnSeqProcessor:
    def __init__(self,channel_order,seq_converter=None, return_stats=True,to_tensor=True):
        if seq_converter is None:
            self._seq_converter = SeqConverter()
        else:
            self._seq_converter = seq_converter
        self._channel_order = channel_order
        self._return_stats = return_stats
        self._to_tensor = to_tensor

    def _validate(self, data):
        if 'answer' in data:
            for id_, input_, answer in zip(data['id'], data['input'],data['answer']):
                seq_length = np.shape(input_)[0]
                ann_length = np.shape(answer)[0]
                if ann_length != seq_length:
                    raise LengthNotEqualException(ann_length, seq_length, id_)

    def _to_dict(self, data):
        has_gene_list = {}
        returned = {'id': [],'input': [],'seq': [],'length': []}
        inputs = self._seq_converter.seqs2dict_vec(data['seq'])
        if 'answer' in data:
            returned['has_gene_status'] = []
            returned['answer'] = []
            ann_seq_dict = genome2dict_vec(data['answer'], self._channel_order)
            for ann_seq in data['answer']:
                has_gene_list[ann_seq.id] = (sum(ann_seq.get_ann('intron')) +
                                             sum(ann_seq.get_ann('exon'))) > 0

        for name in inputs.keys():
            seq = data['seq'][name]
            input_ = inputs[name]
            if self._to_tensor:
                input_ = torch.FloatTensor(input_)
            returned['id'].append(name)
            returned['seq'].append(seq)
            returned['input'].append(input_)
            returned['length'].append(len(seq))
            if 'answer' in data:
                answer = ann_seq_dict[name]
                if self._to_tensor:
                    answer = torch.FloatTensor(answer)
                returned['answer'].append(answer)
                returned['has_gene_status'].append(has_gene_list[name])
            
        return returned

    def process(self, data):
        origin_num = len(data['seq'])
        returned = self._to_dict(data)
        new_num = len(returned['seq'])
        self._validate(returned)
        if self._return_stats:
            stats = {'origin count': origin_num,'filtered count': new_num}
            stats.update(self._get_stats(returned))
            return returned, stats
        else:
            return returned

    def _get_stats(self, data):
        returned = {}
        if 'answer' in data:
            count = {}
            for type_ in self._channel_order:
                count[type_] = 0
            for vec in data['answer']:
                if self._to_tensor:
                    vec = vec.numpy()
                for index,item_count in enumerate(vec.sum(0)):
                    count[self._channel_order[index]] += item_count
            returned['ann_count'] = count
            returned['gene_count'] = int(sum(data['has_gene_status']))
        return returned
