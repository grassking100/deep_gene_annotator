import numpy as np
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.ann_seq_processor import get_background


class RealGenome:
    genome_information = {
        'chromosome': {
            'chr1': 30,
            'chr2': 35
        },
        'source': 'unit_test'
    }
    ANN_TYPES = ['cds', 'intron', 'utr_5', 'utr_3']

    def _create_seq(self, chrom_id, strand_type, seq_id):
        ann_seq = AnnSequence()
        ann_seq.chromosome_id = chrom_id
        ann_seq.ANN_TYPES = RealGenome.ANN_TYPES
        ann_seq.length = RealGenome.genome_information['chromosome'][chrom_id]
        ann_seq.id = seq_id + "_" + strand_type
        ann_seq.strand = strand_type
        ann_seq.source = RealGenome.genome_information['source']
        ann_seq.init_space()
        return ann_seq

    def _create_genome(self, has_background):
        container = AnnSeqContainer()
        ANN_TYPES = RealGenome.ANN_TYPES
        if has_background:
            ANN_TYPES += ['other']
        container.ANN_TYPES = ANN_TYPES
        for id_, length in RealGenome.genome_information['chromosome'].items():
            for strand in ['plus', 'minus']:
                seq = self._create_seq(id_, strand, id_)
                container.add(seq)
        return container

    def _add_seq_to_genome(self, genome, seq):
        chrom = genome.get(seq.chromosome_id + "_" + seq.strand)
        for type_ in seq.ANN_TYPES:
            if seq.strand == 'plus':
                chrom.add_ann(type_, seq.get_ann(type_))
            else:
                chrom.add_ann(type_, np.flip(seq.get_ann(type_), 0))

    def _add_background(self, container):
        focus_types = ['cds', 'intron', 'utr_5', 'utr_3']
        for seq in container:
            background_status = get_background(seq, focus_types)
            seq.set_ann('other', background_status)

    def two_plus_strand(self, has_background):
        container = self._create_genome(has_background)
        self._add_seq_to_genome(container, self._seq2())
        self._add_seq_to_genome(container, self._seq1())
        if has_background:
            self._add_background(container)
        return container

    def two_seq_on_same_plus_strand(self, has_background):
        container = self._create_genome(has_background)
        self._add_seq_to_genome(container, self._seq3())
        self._add_seq_to_genome(container, self._seq1())
        if has_background:
            self._add_background(container)
        return container

    def one_plus_strand_all_utr_5(self, has_background):
        container = self._create_genome(has_background)
        self._add_seq_to_genome(container, self._seq2())
        if has_background:
            self._add_background(container)
        return container

    def one_plus_strand_both_utr(self, has_background):
        container = self._create_genome(has_background)
        self._add_seq_to_genome(container, self._seq1())
        if has_background:
            self._add_background(container)
        return container

    def minus_strand(self, has_background):
        container = self._create_genome(has_background)
        self._add_seq_to_genome(container, self._seq4())
        if has_background:
            self._add_background(container)
        return container

    def multiple_utr_intron_cds(self, has_background):
        container = self._create_genome(has_background)
        self._add_seq_to_genome(container, self._seq6())
        if has_background:
            self._add_background(container)
        return container

    def _seq2(self):
        ann_seq = self._create_seq('chr2', 'plus', 'seq_2')
        ann_seq.set_ann('utr_5', 1, 5, 19)
        return ann_seq

    def _seq6(self):
        ann_seq = self._create_seq('chr2', 'plus', 'seq_6')
        ann_seq.set_ann('utr_5', 1, 5, 6)
        ann_seq.set_ann('intron', 1, 7, 8)
        ann_seq.set_ann('utr_5', 1, 9, 10)
        ann_seq.set_ann('intron', 1, 11, 12)
        ann_seq.set_ann('cds', 1, 13, 14)
        ann_seq.set_ann('intron', 1, 15, 15)
        ann_seq.set_ann('cds', 1, 16, 17)
        ann_seq.set_ann('utr_3', 1, 18, 18)
        ann_seq.set_ann('intron', 1, 19, 19)
        ann_seq.set_ann('utr_3', 1, 20, 20)
        return ann_seq

    def _seq3(self):
        ann_seq = self._create_seq('chr1', 'plus', 'seq_3')
        ann_seq.set_ann('utr_5', 1, 5, 6)
        ann_seq.set_ann('utr_3', 1, 17, 17)
        ann_seq.set_ann('cds', 1, 7, 9)
        ann_seq.set_ann('cds', 1, 15, 16)
        ann_seq.set_ann('intron', 1, 10, 14)
        return ann_seq

    def _seq1(self):
        ann_seq = self._create_seq('chr1', 'plus', 'seq_1')
        ann_seq.set_ann('utr_5', 1, 4, 6)
        ann_seq.set_ann('utr_3', 1, 18, 19)
        ann_seq.set_ann('cds', 1, 7, 10)
        ann_seq.set_ann('cds', 1, 15, 17)
        ann_seq.set_ann('intron', 1, 11, 14)
        return ann_seq

    def _seq4(self):
        ann_seq = self._create_seq('chr1', 'minus', 'seq_4')
        ann_seq.set_ann('utr_3', 1, 29 - 6, 29 - 4)
        ann_seq.set_ann('utr_5', 1, 29 - 19, 29 - 18)
        ann_seq.set_ann('cds', 1, 29 - 10, 29 - 7)
        ann_seq.set_ann('cds', 1, 29 - 17, 29 - 15)
        ann_seq.set_ann('intron', 1, 29 - 14, 29 - 11)
        return ann_seq
