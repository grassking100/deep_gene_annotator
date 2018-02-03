from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.sequence import AnnSequence
class RealGenome:
    def two_plus_strand(self,has_intergenic_region):
        container = AnnSeqContainer()
        container.ANN_TYPES = ['intergenic_region','cds','intron','utr_5','utr_3']
        container.add(self._seq2(has_intergenic_region))
        container.add(self._seq1(has_intergenic_region))
        return container
    def two_seq_on_same_plus_strand(self,has_intergenic_region):
        container = AnnSeqContainer()
        container.ANN_TYPES = ['intergenic_region','cds','intron','utr_5','utr_3']
        container.add(self._seq3(has_intergenic_region))
        container.add(self._seq1(has_intergenic_region))
        return container
    def one_plus_strand_all_utr_5(self, has_intergenic_region):
        container = AnnSeqContainer()
        container.ANN_TYPES = ['intergenic_region','cds','intron','utr_5','utr_3']
        container.add(self._seq2(has_intergenic_region))
        return container
    def one_plus_strand_both_utr(self, has_intergenic_region):
        container = AnnSeqContainer()
        container.ANN_TYPES = ['intergenic_region','cds','intron','utr_5','utr_3']
        container.add(self._seq1(has_intergenic_region))
        return container
    def _seq2(self, has_intergenic_region):
        ann_seq = AnnSequence()
        ann_seq.chromosome_id='chr2'
        ann_seq.strand='plus'
        ann_seq.length=35
        ann_seq.id = 'seq_2'
        ann_seq.source = 'unit_test'
        ann_seq.ANN_TYPES = ['intergenic_region','cds','intron','utr_5','utr_3']
        ann_seq.initSpace()
        ann_seq.set_ann('utr_5',1,5,19)
        if has_intergenic_region:
            ann_seq.set_ann('intergenic_region',1,0,4)
            ann_seq.set_ann('intergenic_region',1,20,29)
        return ann_seq
    def _seq3(self, has_intergenic_region):
        ann_seq = AnnSequence()
        ann_seq.chromosome_id='chr1'
        ann_seq.strand='plus'
        ann_seq.length=30
        ann_seq.id = 'seq_3'
        ann_seq.source = 'unit_test'
        ann_seq.ANN_TYPES = ['intergenic_region','cds','intron','utr_5','utr_3']
        ann_seq.initSpace()
        ann_seq.set_ann('utr_5',1,5,6)
        ann_seq.set_ann('utr_3',1,17,17)
        ann_seq.set_ann('cds',1,7,9)
        ann_seq.set_ann('cds',1,15,16)
        ann_seq.set_ann('intron',1,10,14)
        if has_intergenic_region:
            ann_seq.set_ann('intergenic_region',1,0,4)
            ann_seq.set_ann('intergenic_region',1,18,29)
        return ann_seq
    def _seq1(self, has_intergenic_region):
        ann_seq = AnnSequence()
        ann_seq.chromosome_id='chr1'
        ann_seq.strand='plus'
        ann_seq.length=30
        ann_seq.id = 'seq_1'
        ann_seq.source = 'unit_test'
        ann_seq.ANN_TYPES = ['intergenic_region','cds','intron','utr_5','utr_3']
        ann_seq.initSpace()
        ann_seq.set_ann('utr_5',1,5,6)
        ann_seq.set_ann('utr_3',1,18,19)
        ann_seq.set_ann('cds',1,7,9)
        ann_seq.set_ann('cds',1,15,17)
        ann_seq.set_ann('intron',1,10,14)
        if has_intergenic_region:
            ann_seq.set_ann('intergenic_region',1,0,4)
            ann_seq.set_ann('intergenic_region',1,20,29)
        return ann_seq
    def _seq4(self, has_intergenic_region):
        ann_seq = AnnSequence()
        ann_seq.chromosome_id='chr1'
        ann_seq.strand='minus'
        ann_seq.length=30
        ann_seq.id = 'seq_4'
        ann_seq.source = 'unit_test'
        ann_seq.ANN_TYPES = ['intergenic_region','cds','intron','utr_5','utr_3']
        ann_seq.initSpace()
        ann_seq.set_ann('utr_5',1,5,6)
        ann_seq.set_ann('utr_3',1,18,19)
        ann_seq.set_ann('cds',1,7,9)
        ann_seq.set_ann('cds',1,15,17)
        ann_seq.set_ann('intron',1,10,14)
        if has_intergenic_region:
            ann_seq.set_ann('intergenic_region',1,0,4)
            ann_seq.set_ann('intergenic_region',1,20,29)
        return ann_seq