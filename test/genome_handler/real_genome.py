import unittest
import numpy as np
from . import AnnSeqContainer
from . import AnnSequence
class RealGenome:
    genome_information={'chromosome':{'chr1':30,'chr2':35},'source':'unit_test'}
    ANN_TYPES = ['intergenic_region','cds','intron','utr_5','utr_3']
    def _create_seq(self,chrom_id,strand_type,seq_id):
        ann_seq = AnnSequence()
        ann_seq.chromosome_id=chrom_id
        ann_seq.ANN_TYPES = RealGenome.ANN_TYPES
        ann_seq.length=RealGenome.genome_information['chromosome'][chrom_id]
        ann_seq.id = seq_id+"_"+strand_type
        ann_seq.strand = strand_type
        ann_seq.source = RealGenome.genome_information['source']
        ann_seq.initSpace()
        return ann_seq
    def _create_genome(self):
        container = AnnSeqContainer()
        container.ANN_TYPES = RealGenome.ANN_TYPES
        for id_, length in RealGenome.genome_information['chromosome'].items():
            for strand in ['plus','minus']:
                seq = self._create_seq(id_,strand,id_)
                container.add(seq)
        return container
    def _add_seq_to_genome(self, genome, seq):
        chrom = genome.get(seq.chromosome_id+"_"+seq.strand)
        for type_ in seq.ANN_TYPES:
            chrom.add_ann(type_,seq.get_ann(type_))
    def two_plus_strand(self,has_intergenic_region):
        container = self._create_genome()
        self._add_seq_to_genome(container, self._seq2(has_intergenic_region))
        self._add_seq_to_genome(container, self._seq1(has_intergenic_region))
        return container
    def two_seq_on_same_plus_strand(self,has_intergenic_region):
        container = self._create_genome()
        self._add_seq_to_genome(container, self._seq3(has_intergenic_region))
        self._add_seq_to_genome(container, self._seq1(has_intergenic_region))
        return container
    def one_plus_strand_all_utr_5(self, has_intergenic_region):
        container = self._create_genome()
        self._add_seq_to_genome(container, self._seq2(has_intergenic_region))
        return container
    def one_plus_strand_both_utr(self, has_intergenic_region):
        container =self._create_genome()
        self._add_seq_to_genome(container, self._seq1(has_intergenic_region))
        return container
    def minus_strand(self,has_intergenic_region):
        container =self._create_genome()
        self._add_seq_to_genome(container, self._seq4(has_intergenic_region))
        return container
    def multiple_utr_intron_cds(self, has_intergenic_region):
        container = self._create_genome()
        self._add_seq_to_genome(container, self._seq6(has_intergenic_region))
        return container
    def _seq2(self, has_intergenic_region):
        ann_seq = self._create_seq('chr2','plus','seq_2')
        ann_seq.set_ann('utr_5',1,5,19)
        if has_intergenic_region:
            ann_seq.set_ann('intergenic_region',1,0,4)
            ann_seq.set_ann('intergenic_region',1,20,29)
        return ann_seq
    def _seq6(self, has_intergenic_region):
        ann_seq = self._create_seq('chr2','plus','seq_6')
        ann_seq.set_ann('utr_5',1,5,6)
        ann_seq.set_ann('intron',1,7,8)
        ann_seq.set_ann('utr_5',1,9,10)
        ann_seq.set_ann('intron',1,11,12)
        ann_seq.set_ann('cds',1,13,14)
        ann_seq.set_ann('intron',1,15,15)
        ann_seq.set_ann('cds',1,16,17)
        ann_seq.set_ann('utr_3',1,18,18)
        ann_seq.set_ann('intron',1,19,19)
        ann_seq.set_ann('utr_3',1,20,20)
        if has_intergenic_region:
            ann_seq.set_ann('intergenic_region',1,0,4)
            ann_seq.set_ann('intergenic_region',1,21,29)
        return ann_seq
    def _seq3(self, has_intergenic_region):
        ann_seq = self._create_seq('chr1','plus','seq_3')
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
        ann_seq = self._create_seq('chr1','plus','seq_1')
        ann_seq.set_ann('utr_5',1,4,6)
        ann_seq.set_ann('utr_3',1,18,19)
        ann_seq.set_ann('cds',1,7,10)
        ann_seq.set_ann('cds',1,15,17)
        ann_seq.set_ann('intron',1,11,14)
        if has_intergenic_region:
            ann_seq.set_ann('intergenic_region',1,0,3)
            ann_seq.set_ann('intergenic_region',1,20,29)
        return ann_seq
    def _seq4(self, has_intergenic_region):
        ann_seq = self._create_seq('chr1','minus','seq_4')
        ann_seq.set_ann('utr_3',1,29-6,29-4)
        ann_seq.set_ann('utr_5',1,29-19,29-18)
        ann_seq.set_ann('cds',1,29-10,29-7)
        ann_seq.set_ann('cds',1,29-17,29-15)
        ann_seq.set_ann('intron',1,29-14,29-11)
        if has_intergenic_region:
            ann_seq.set_ann('intergenic_region',1,29-3,29-0)
            ann_seq.set_ann('intergenic_region',1,29-29,29-20)
        return ann_seq