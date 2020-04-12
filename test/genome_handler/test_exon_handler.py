from ..ann_seq_test_case import AnnSeqTestCase
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.exon_handler import ExonHandler


class TestExonHandler(AnnSeqTestCase):
    def test_further_division_exon(self):
        exon_handler = ExonHandler()
        ann_seq = AnnSequence()
        ann_seq.ANN_TYPES = ['utr_5', 'utr_3', 'cds', 'intron', 'other']
        ann_seq.length = 10
        ann_seq.id = 'input'
        ann_seq.strand = 'plus'
        ann_seq.chromosome_id = 'test'
        ann_seq.init_space()
        ann_seq.set_ann('other', 1, 0, 0).set_ann('utr_5', 1, 1,
                                                  1).set_ann('cds', 1, 2, 2)
        ann_seq.set_ann('intron', 1, 3,
                        4).set_ann('cds', 1, 5, 5).set_ann('intron', 1, 6, 6)
        ann_seq.set_ann('utr_3', 1, 7, 7).set_ann('intron', 1, 8,
                                                  8).set_ann('utr_3', 1, 9, 9)
        further_division_seq = exon_handler.further_division(ann_seq)
        expected_seq = AnnSequence()
        expected_seq.ANN_TYPES = exon_handler.internal_subtypes + \
            exon_handler.external_subtypes + ['other', 'intron']
        expected_seq.length = 10
        expected_seq.id = 'input'
        expected_seq.strand = 'plus'
        expected_seq.chromosome_id = 'test'
        expected_seq.init_space()
        expected_seq.set_ann('other', 1, 0,
                             0).set_ann('external_utr_5', 1, 1,
                                        1).set_ann('internal_cds', 1, 2, 2)
        expected_seq.set_ann('intron', 1, 3,
                             4).set_ann('internal_cds', 1, 5,
                                        5).set_ann('intron', 1, 6, 6)
        expected_seq.set_ann('internal_utr_3', 1, 7,
                             7).set_ann('intron', 1, 8,
                                        8).set_ann('external_utr_3', 1, 9, 9)
        expected_seq.processed_status = 'further_division'
        self.assert_seq_equal(expected_seq, further_division_seq)

    def test_simplify_exon_name(self):
        exon_handler = ExonHandler()
        expected_seq = AnnSequence()
        expected_seq.ANN_TYPES = ['utr_5', 'utr_3', 'cds', 'intron', 'other']
        expected_seq.processed_status = 'one_hot'
        expected_seq.length = 10
        expected_seq.id = 'input'
        expected_seq.strand = 'plus'
        expected_seq.chromosome_id = 'test'
        expected_seq.init_space()
        expected_seq.set_ann('other', 1, 0,
                             0).set_ann('utr_5', 1, 1,
                                        2).set_ann('intron', 1, 3, 4)
        expected_seq.set_ann('cds', 1, 5,
                             5).set_ann('intron', 1, 6,
                                        6).set_ann('utr_3', 1, 7, 7)
        expected_seq.set_ann('intron', 1, 8, 8).set_ann('utr_3', 1, 9, 9)
        ann_seq = AnnSequence()
        ann_seq.ANN_TYPES = exon_handler.internal_subtypes + \
            exon_handler.external_subtypes + ['other', 'intron']
        ann_seq.length = 10
        ann_seq.processed_status = 'one_hot'
        ann_seq.id = 'input'
        ann_seq.strand = 'plus'
        ann_seq.chromosome_id = 'test'
        ann_seq.init_space()
        ann_seq.set_ann('other', 1, 0,
                        0).set_ann('external_utr_5', 1, 1,
                                   2).set_ann('intron', 1, 3, 4)
        ann_seq.set_ann('internal_cds', 1, 5,
                        5).set_ann('intron', 1, 6,
                                   6).set_ann('internal_utr_3', 1, 7, 7)
        ann_seq.set_ann('intron', 1, 8, 8).set_ann('external_utr_3', 1, 9, 9)
        simplify_exon_seq = exon_handler.simplify_exon_name(ann_seq)
        self.assert_seq_equal(expected_seq, simplify_exon_seq)
