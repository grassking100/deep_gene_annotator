import numpy as np
from ..genome_handler.ann_genome_processor import genome2dict_vec
from ..utils.seq_converter import SeqConverter
from ..utils.exception import LengthNotEqualException


class AnnSeqProcessor:
    def __init__(self,channel_order,seq_converter=None):
        if seq_converter is None:
            self._seq_converter = SeqConverter()
        else:
            self._seq_converter = seq_converter
        self._channel_order = channel_order

    def _validate(self, data):
        if 'answers' in data:
            for id_, input_, answer in zip(data['ids'], data['inputs'],data['answers']):
                seq_length = np.shape(input_)[0]
                ann_length = np.shape(answer)[0]
                if ann_length != seq_length:
                    raise LengthNotEqualException(ann_length, seq_length, id_)

    def _to_dict(self, item):
        data = {'ids': [],'inputs': [],'seqs': [],'lengths': [],'has_gene_statuses': []}
        has_gene_list = {}
        seqs = self._seq_converter.seqs2dict_vec(item['inputs'])
        if 'answers' in item:
            data['answers'] = []
            ann_seq_dict = genome2dict_vec(item['answers'], self._channel_order)
            for ann_seq in item['answers']:
                has_gene_list[ann_seq.id] = (sum(ann_seq.get_ann('intron')) +
                                             sum(ann_seq.get_ann('exon'))) > 0

        for name in seqs.keys():
            seq = seqs[name]
            data['ids'].append(name)
            data['seqs'].append(item['inputs'][name])
            data['inputs'].append(seq)
            data['lengths'].append(len(seq))
            if 'answers' in item:
                answer = ann_seq_dict[name]
                data['answers'].append(answer)
                data['has_gene_statuses'].append(has_gene_list[name])
            
        return data

    def process(self, data, return_stats=False):
        returned = {}
        stats = {}
        for purpose in data.keys():
            item = data[purpose]
            origin_num = len(item['inputs'])
            item = self._to_dict(item)
            new_num = len(item['inputs'])
            self._validate(item)
            returned[purpose] = item
            if return_stats:
                stats[purpose] = {'origin count': origin_num,'filtered count': new_num}
                stats[purpose].update(self._get_stats(item))
        if return_stats:
            return returned, stats
        else:
            return returned

    def _get_stats(self, data):
        returned = {}
        if 'answers' in data:
            count = {}
            for type_ in self._channel_order:
                count[type_] = 0
            for vec in data['answers']:
                for index,item_count in enumerate(vec.sum(0)):
                    count[self._channel_order[index]] += item_count
            returned['ann_count'] = count
            returned['gene_count'] = int(sum(data['has_gene_statuses']))
        return returned
