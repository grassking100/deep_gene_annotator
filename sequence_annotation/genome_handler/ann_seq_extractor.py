from abc import ABCMeta
from .sequence import AnnSequence
from .seq_container import AnnSeqContainer

class AnnSeqExtractor(metaclass=ABCMeta):
    def extract(self,ann_seq_container,seq_info_container):
        result = AnnSeqContainer()
        result.ANN_TYPES = ann_seq_container.ANN_TYPES
        for seq_info in seq_info_container.data:
            one_strand_chromosome = ann_seq_container.get(str(seq_info.chromosome_id)+"_"+seq_info.strand)
            seq_ann=self._extract_seq(one_strand_chromosome,seq_info)
            result.add(seq_ann) 
        return result
    def _extract_seq(self,one_strand_chromosome,seq_info):
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
