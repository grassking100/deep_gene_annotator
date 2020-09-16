import os
import unittest
from sequence_annotation.file_process.utils import InvalidStrandType
from sequence_annotation.file_process.seq_info_parser import BedInfoParser

bed_example_root = os.path.join(os.path.dirname(__file__), '..','data','bed')


class TestSeqInfoParser(unittest.TestCase):
    def test_bed_test_two_strand(self):
        file_path = os.path.join(bed_example_root, 'two_plus_strand.bed')
        parser = BedInfoParser()
        result = parser.parse(file_path)
        self.assertEqual(list, type(result))

    def test_bed_read_get_data_type(self):
        file_path = os.path.join(bed_example_root,'one_plus_strand_both_utr.bed')
        parser = BedInfoParser()
        result = parser.parse(file_path)
        self.assertEqual(list, type(result))

    def test_bed_negative_index(self):
        file_path = os.path.join(bed_example_root, 'negative_index.bed')
        parser = BedInfoParser()
        with self.assertRaises(Exception):
            parser.parse(file_path)

    def test_bed_invalid_strand(self):
        file_path = os.path.join(bed_example_root, 'invalid_strand.bed')
        parser = BedInfoParser()
        with self.assertRaises(InvalidStrandType):
            parser.parse(file_path)
