import warnings
import random
import numpy as np
from ..genome_handler.seq_container import AnnSeqContainer
from ..genome_handler import ann_genome_processor
from ..utils.seq_converter import SeqConverter
from ..utils.exception import LengthNotEqualException
from ..utils.utils import get_subdict


class AnnSeqProcessor:
    def __init__(self,
                 channel_order,
                 seq_converter=None,
                 validation_split=None):
        self._validation_split = validation_split or 0
        if seq_converter is None:
            self._seq_converter = SeqConverter()
        else:
            self._seq_converter = seq_converter
        self._channel_order = channel_order

    def _validate(self, data):
        if 'answers' in data:
            for id_, input_, answer in zip(data['ids'], data['inputs'],
                                           data['answers']):
                seq_length = np.shape(input_)[0]
                ann_length = np.shape(answer)[0]
                if ann_length != seq_length:
                    raise LengthNotEqualException(ann_length, seq_length, id_)

    def _split(self, data):
        if self._validation_split > 0 and not 'validation' in data.keys():
            returned = {}
            shuffled_keys = list(data['training']['inputs'].keys())
            random.shuffle(shuffled_keys)
            val_length = int(len(shuffled_keys) * self._validation_split)
            train_keys = shuffled_keys[val_length:]
            val_keys = shuffled_keys[:val_length]
            train_seqs = {}
            val_seqs = {}
            for type_, item in data.items():
                if isinstance(item, AnnSeqContainer):
                    train_seqs[type_] = item.get_seqs(train_keys)
                    val_seqs[type_] = item.get_seqs(val_keys)
                else:
                    train_seqs[type_] = get_subdict(train_keys, item)
                    val_seqs[type_] = get_subdict(val_keys, item)
            returned['training'] = train_seqs
            returned['validation'] = val_seqs
        else:
            returned = data
        return returned

    def _to_dict(self, item):
        data = {
            'inputs': [],
            'lengths': [],
            'ids': [],
            'seqs': [],
            'has_gene_statuses': []
        }
        has_gene_list = {}
        seqs = self._seq_converter.seqs2dict_vec(item['inputs'])
        if 'answers' in item:
            data['answers'] = []
            ann_seq_dict = ann_genome_processor.genome2dict_vec(
                item['answers'], self._channel_order)

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

    def process(self, data):
        splitted_data = self._split(data)
        returned = {}
        warning = "{} data have {} sequences, it left {} sequences after filtering"
        for purpose in splitted_data.keys():
            item = splitted_data[purpose]
            origin_num = len(item['inputs'])
            item = self._to_dict(item)
            new_num = len(item['inputs'])
            if origin_num != new_num:
                warnings.warn(warning.format(purpose, origin_num, new_num),
                              UserWarning)
            self._validate(item)
            returned[purpose] = item
        return returned
