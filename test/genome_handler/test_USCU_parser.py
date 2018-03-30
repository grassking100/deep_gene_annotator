import unittest
from . import InvalidStrandType,NegativeNumberException
from . import ReturnNoneException
from . import USCUParser
from . import file_prefix
class TestUSCUParser(unittest.TestCase):
    def test_test_two_strand(self):
        file_path = file_prefix + 'two_plus_strand.tsv'
        parser = USCUParser(file_path)
        parser.parse()
        self.assertEqual(list,type(parser.result))
    def test_read_get_data_type(self):
        file_path = file_prefix + 'one_plus_strand_both_utr.tsv'
        parser = USCUParser(file_path)
        parser.parse()
        self.assertEqual(list,type(parser.result))
    def test_not_parse(self):
        file_path = file_prefix + 'one_plus_strand_both_utr.tsv'
        parser = USCUParser(file_path)
        with self.assertRaises(ReturnNoneException):
            data = parser.result
    def test_negative_index(self):
        file_path = file_prefix + 'negative_index.tsv'
        parser = USCUParser(file_path)
        self.assertRaises(NegativeNumberException,parser.parse)
    def test_invalid_strand(self):
        file_path = file_prefix + 'invalid_strand.tsv'
        parser = USCUParser(file_path)
        self.assertRaises(InvalidStrandType,parser.parse)
if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestUSCUParser)
    unittest.main()

