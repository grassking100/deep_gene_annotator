import os
import unittest
import numpy as np
from sequence_annotation.file_process.seq_info_parser import BedInfoParser
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.ann_seq_converter import CodingBedSeqConverter
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from .real_genome import RealGenome

bed_example_root = os.path.join(os.path.dirname(__file__), '..','data','bed')

def _create_genome(path):
    parser = BedInfoParser()
    converter = CodingBedSeqConverter()
    parsed_seqs = parser.parse(path)
    ann_seqs = []
    for parsed_seq in parsed_seqs:
        ann_seq = converter.convert(parsed_seq)
        ann_seqs.append(ann_seq)
    container = AnnSeqContainer()
    container.ANN_TYPES = ann_seqs[0].ANN_TYPES
    container.add(ann_seqs)
    creator = AnnGenomeCreator()
    result = creator.create(container, RealGenome.genome_information)
    return result


class TestAnnGenomeCreator(unittest.TestCase):
    real_genome = RealGenome()

    def test_create_type(self):
        file_path = os.path.join(bed_example_root,'one_plus_strand_all_utr_5.bed')
        result = _create_genome(file_path)
        self.assertEqual(AnnSeqContainer, type(result))

    def _test_seq(self, real_seq, test_seq):
        self.assertEqual(real_seq.id, test_seq.id)
        self.assertEqual(real_seq.strand, test_seq.strand)
        self.assertEqual(real_seq.length, test_seq.length)
        self.assertEqual(real_seq.source, test_seq.source)
        for type_ in real_seq.ANN_TYPES:
            np.testing.assert_array_equal(real_seq.get_ann(type_),
                                          test_seq.get_ann(type_),
                                          err_msg="Wrong type:" + type_ + "(" +
                                          str(real_seq.id) + ")")

    def _test_genome(self, real_genome, test_genome):
        self.assertEqual(set(real_genome.ANN_TYPES),
                         set(test_genome.ANN_TYPES))
        self.assertEqual(len(real_genome.data), len(test_genome.data))
        for real_seq, test_seq in zip(real_genome.data, test_genome.data):
            self._test_seq(real_seq, test_seq)

    def test_one_plus_strand_all_utr_5(self):
        file_path = os.path.join(bed_example_root,'one_plus_strand_all_utr_5.bed')
        real_genome = TestAnnGenomeCreator.real_genome.one_plus_strand_all_utr_5(
            False)
        result = _create_genome(file_path)
        self._test_genome(real_genome, result)

    def test_two_plus_strand(self):
        file_path = os.path.join(bed_example_root, 'two_plus_strand.bed')
        real_genome = TestAnnGenomeCreator.real_genome.two_plus_strand(False)
        result = _create_genome(file_path)
        self._test_genome(real_genome, result)

    def test_two_seq_on_same_plus(self):
        file_path = os.path.join(bed_example_root,'two_seq_on_same_plus_strand.bed')
        real_genome = TestAnnGenomeCreator.real_genome.two_seq_on_same_plus_strand(
            False)
        result = _create_genome(file_path)
        self._test_genome(real_genome, result)

    def test_one_plus_strand_both(self):
        file_path = os.path.join(bed_example_root,'one_plus_strand_both_utr.bed')
        real_genome = TestAnnGenomeCreator.real_genome.one_plus_strand_both_utr(
            False)
        result = _create_genome(file_path)
        self._test_genome(real_genome, result)

    def test_minus_strand(self):
        file_path = os.path.join(bed_example_root, 'minus_strand.bed')
        real_genome = TestAnnGenomeCreator.real_genome.minus_strand(False)
        result = _create_genome(file_path)
        self._test_genome(real_genome, result)

    def test_multiple_utr_intron_cds(self):
        file_path = os.path.join(bed_example_root,'multiple_utr_intron_cds.bed')
        real_genome = TestAnnGenomeCreator.real_genome.multiple_utr_intron_cds(
            False)
        result = _create_genome(file_path)
        self._test_genome(real_genome, result)
