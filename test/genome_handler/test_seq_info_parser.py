import unittest
import pandas as pd
from . import InvalidStrandType,NegativeNumberException
from . import UscuInfoParser
from . import EnsemblInfoParser
from . import ucsc_file_prefix
from . import ensembl_file_prefix
class TestSeqInfoParser(unittest.TestCase):
    def test_USCU_test_two_strand(self):
        file_path = ucsc_file_prefix + 'two_plus_strand.tsv'
        data = pd.read_csv(file_path,sep='\t').to_dict('record')
        parser = UscuInfoParser()
        result = parser.parse(data)
        self.assertEqual(list,type(result))
    def test_USCU_read_get_data_type(self):
        file_path = ucsc_file_prefix + 'one_plus_strand_both_utr.tsv'
        data = pd.read_csv(file_path,sep='\t').to_dict('record')
        parser = UscuInfoParser()
        result = parser.parse(data)
        self.assertEqual(list,type(result))
    def test_USCU_negative_index(self):
        file_path = ucsc_file_prefix + 'negative_index.tsv'
        data = pd.read_csv(file_path,sep='\t').to_dict('record')
        parser = UscuInfoParser()
        with self.assertRaises(NegativeNumberException):
            parser.parse(data)
    def test_USCU_invalid_strand(self):
        file_path = ucsc_file_prefix + 'invalid_strand.tsv'
        data = pd.read_csv(file_path,sep='\t').to_dict('record')
        parser = UscuInfoParser()
        with self.assertRaises(InvalidStrandType):
            parser.parse(data)
    def test_ensembl_test_ENSTNIP00000005172(self):
        file_path = ensembl_file_prefix + 'ENSTNIP00000005172.tsv'
        data = pd.read_csv(file_path,sep='\t').to_dict('record')
        parser = EnsemblInfoParser()
        result = parser.parse(data)
        self.assertEqual(list,type(result))
    def test_ensembl_negative_index(self):
        file_path = ensembl_file_prefix + 'negative_index_ENSTNIP00000005172.tsv'
        data = pd.read_csv(file_path,sep='\t').to_dict('record')
        parser = EnsemblInfoParser()
        with self.assertRaises(NegativeNumberException):
            parser.parse(data)
    def test_ensembl_invalid_strand(self):
        file_path = ensembl_file_prefix + 'invalid_strand_ENSTNIP00000005172.tsv'
        data = pd.read_csv(file_path,sep='\t').to_dict('record')
        parser = EnsemblInfoParser()
        with self.assertRaises(InvalidStrandType):
            parser.parse(data)
            
if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestSeqInfoParser)
    unittest.main()

