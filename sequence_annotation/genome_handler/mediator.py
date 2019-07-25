import pandas as pd
from .seq_container import AnnSeqContainer

class Mediator:
    def __init__(self,seq_info_parser,ann_seq_converter):
        self._parser = seq_info_parser
        self._converter = ann_seq_converter

    def create(self,path):
        ann_seqs = AnnSeqContainer(self._converter.ANN_TYPES)
        parsed = self._parser.parse(path)
        for item in parsed:
            ann_seq = self._converter.convert(item)
            ann_seqs.add(ann_seq)
        return ann_seqs
