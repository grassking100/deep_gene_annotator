import unittest
import sys
sys.path.append("../../")
from sequence_annotation.utils.exception import InvalidStrandType,NegativeNumberException,ReturnNoneException
from sequence_annotation.genome_handler.seq_info_parser import USCUParser
class TestUSCUParser(unittest.TestCase):
    def test_test_two_strand(self):
        file_path = 'data/ucsc/two_plus_strand.tsv'
        parser = USCUParser(file_path)
        parser.parse()
        self.assertEqual(list,type(parser.data))
    def test_read_get_data_type(self):
        file_path = 'data/ucsc/one_plus_strand_both_utr.tsv'
        parser = USCUParser(file_path)
        parser.parse()
        self.assertEqual(list,type(parser.data))
    def test_not_parse(self):
        file_path = 'data/ucsc/one_plus_strand_both_utr.tsv'
        parser = USCUParser(file_path)
        with self.assertRaises(ReturnNoneException):
            data = parser.data
    def test_negative_index(self):
        file_path = 'data/ucsc/negative_index.tsv'
        parser = USCUParser(file_path)
        self.assertRaises(NegativeNumberException,parser.parse)
    def test_invalid_strand(self):
        file_path = 'data/ucsc/invalid_strand.tsv'
        parser = USCUParser(file_path)
        self.assertRaises(InvalidStrandType,parser.parse)       
if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestUSCUParser)
    unittest.main()

