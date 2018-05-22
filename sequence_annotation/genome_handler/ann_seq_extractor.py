from abc import ABCMeta
from abc import abstractmethod
from . import AnnSequence
from . import AnnSeqContainer
class AnnSeqExtractor(metaclass=ABCMeta):
    def extract(self,ann_seq_container,seq_info_container):
        return self._extract_seqs(seq_info_container,ann_seq_container)
    def _extract_seq(self,seq_info,one_strand_chromosome):
        ann_seq = AnnSequence()
        ann_seq.ANN_TYPES = one_strand_chromosome.ANN_TYPES
        ann_seq.length=seq_info.length
        ann_seq.id=seq_info.id
        ann_seq.chromosome_id=seq_info.chromosome_id
        ann_seq.strand=seq_info.strand
        ann_seq.source=seq_info.source
        ann_seq.init_space()
        for type_ in one_strand_chromosome.ANN_TYPES:
            ann = one_strand_chromosome.get_ann(type_,seq_info.start,seq_info.end)
            ann_seq.add_ann(type_,ann)
        return ann_seq
    def _extract_seqs(self,seq_info_container,ann_seq_container):
        result = AnnSeqContainer()
        result.ANN_TYPES = ann_seq_container.ANN_TYPES
        for seq_info in seq_info_container.data:
            one_strand_chromosome = ann_seq_container.get(str(seq_info.chromosome_id)+"_"+seq_info.strand)
            seq_ann=self._extract_seq(seq_info,one_strand_chromosome)
            result.add(seq_ann)
        return result