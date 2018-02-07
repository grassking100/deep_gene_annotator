from . import AnnSequence
from . import AnnSeqContainer
from . import validate_return
class AnnSeqExtractor:
    def __init__(self,ann_seq_container,seq_info_container):
        self._seq_info_container=seq_info_container
        self._ann_seq_container=ann_seq_container
        self._result=AnnSeqContainer()
        self._result.ANN_TYPES=self._ann_seq_container.ANN_TYPES
    def extract(self):
        self._extract_seqs()
    @property
    @validate_return("use method extract before access the data")
    def result(self):
        return self._result
    def _validate(self):
        pass
    @property
    def selected_sequences_annotation(self):
        return self.__selected_sequences_annotation
    def _get_extract_seq(self,seq_info):
        ann_seq = AnnSequence()
        ann_seq.ANN_TYPES = self._result.ANN_TYPES
        ann_seq.length=seq_info.length
        ann_seq.id=seq_info.id
        ann_seq.chromosome_id=seq_info.chromosome_id
        ann_seq.strand=seq_info.strand
        ann_seq.source=seq_info.source
        ann_seq.initSpace()
        chrom = self._ann_seq_container.get(seq_info.chromosome_id+"_"+seq_info.strand)
        for type_ in ann_seq.ANN_TYPES:
            ann = chrom.get_ann(type_,seq_info.start,seq_info.end)
            ann_seq.add_ann(type_,ann)
        return ann_seq
    def _extract_seqs(self):
        for seq_info in self._seq_info_container.data:
            seq_ann=self._get_extract_seq(seq_info)
            self._result.add(seq_ann)