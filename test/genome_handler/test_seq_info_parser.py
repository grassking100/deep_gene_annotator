import os
import unittest
from sequence_annotation.utils.exception import InvalidStrandType
from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser

file_root = os.path.abspath(os.path.join(__file__, '..'))
bed_file_prefix = os.path.join(file_root, 'data/bed')


class TestSeqInfoParser(unittest.TestCase):
    def test_bed_test_two_strand(self):
        file_path = os.path.join(bed_file_prefix, 'two_plus_strand.bed')
        parser = BedInfoParser()
        result = parser.parse(file_path)
        self.assertEqual(list, type(result))

    def test_bed_read_get_data_type(self):
        file_path = os.path.join(bed_file_prefix,
                                 'one_plus_strand_both_utr.bed')
        parser = BedInfoParser()
        result = parser.parse(file_path)
        self.assertEqual(list, type(result))

    def test_bed_negative_index(self):
        file_path = os.path.join(bed_file_prefix, 'negative_index.bed')
        parser = BedInfoParser()
        with self.assertRaises(Exception):
            parser.parse(file_path)

    def test_bed_invalid_strand(self):
        file_path = os.path.join(bed_file_prefix, 'invalid_strand.bed')
        parser = BedInfoParser()
        with self.assertRaises(InvalidStrandType):
            parser.parse(file_path)
