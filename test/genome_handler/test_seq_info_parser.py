import unittest
import pandas as pd
from sequence_annotation.utils.exception import NegativeNumberException
from sequence_annotation.genome_handler.exception import InvalidStrandType
from sequence_annotation.genome_handler.seq_info_parser import UCSCInfoParser,EnsemblInfoParser
from . import ucsc_file_prefix
from . import ensembl_file_prefix
class TestSeqInfoParser(unittest.TestCase):
    def test_USCU_test_two_strand(self):
        file_path = ucsc_file_prefix + 'two_plus_strand.tsv'
        parser = UCSCInfoParser()
        result = parser.parse(file_path)
        self.assertEqual(list,type(result))
    def test_USCU_read_get_data_type(self):
        file_path = ucsc_file_prefix + 'one_plus_strand_both_utr.tsv'
        parser = UCSCInfoParser()
        result = parser.parse(file_path)
        self.assertEqual(list,type(result))
    def test_USCU_negative_index(self):
        file_path = ucsc_file_prefix + 'negative_index.tsv'
        parser = UCSCInfoParser()
        with self.assertRaises(NegativeNumberException):
            parser.parse(file_path)
    def test_USCU_invalid_strand(self):
        file_path = ucsc_file_prefix + 'invalid_strand.tsv'
        parser = UCSCInfoParser()
        with self.assertRaises(InvalidStrandType):
            parser.parse(file_path)
    def test_ensembl_test_ENSTNIP00000005172(self):
        file_path = ensembl_file_prefix + 'sorted_merged_ensembl_tetraodon_8_0_ENSTNIP00000005172.tsv'
        parser = EnsemblInfoParser()
        result = parser.parse(file_path)
        self.assertEqual(list,type(result))
    def test_ensembl_negative_index(self):
        file_path = ensembl_file_prefix + 'negative_index_ENSTNIP00000005172.tsv'
        parser = EnsemblInfoParser()
        with self.assertRaises(NegativeNumberException):
            parser.parse(file_path)
    def test_ensembl_invalid_strand(self):
        file_path = ensembl_file_prefix + 'invalid_strand_ENSTNIP00000005172.tsv'
        parser = EnsemblInfoParser()
        with self.assertRaises(InvalidStrandType):
            parser.parse(file_path)
    def test_ensembl_invalid_strand(self):
        file_path = ensembl_file_prefix + 'invalid_strand_ENSTNIP00000005172.tsv'
        parser = EnsemblInfoParser()
        with self.assertRaises(InvalidStrandType):
            parser.parse(file_path)
