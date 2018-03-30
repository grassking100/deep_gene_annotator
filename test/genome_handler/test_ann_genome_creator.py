import unittest
import numpy as np
from . import AnnSeqContainer
from . import USCUParser
from . import AnnGenomeCreator
from . import ReturnNoneException
from . import RealGenome
from . import file_prefix
class TestAnnGenomeCreator(unittest.TestCase):
    real_genome = RealGenome()
    def test_not_create(self):
        file_path =file_prefix + 'one_plus_strand_all_utr_5.tsv'
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(RealGenome.genome_information,parser.result)
        with self.assertRaises(ReturnNoneException):
            data = creator.result
    def test_create_type(self):
        file_path = file_prefix + 'one_plus_strand_all_utr_5.tsv'
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(RealGenome.genome_information,parser.result)
        creator.create()
        self.assertEqual(AnnSeqContainer,type(creator.result))
    def _test_seq(self, real_seq,test_seq):
        self.assertEqual(real_seq.id,test_seq.id)
        self.assertEqual(real_seq.strand,test_seq.strand)
        self.assertEqual(real_seq.length,test_seq.length)
        self.assertEqual(real_seq.source,test_seq.source)
        for type_ in real_seq.ANN_TYPES:
            np.testing.assert_array_equal(real_seq.get_ann(type_),
                                          test_seq.get_ann(type_),
                                          err_msg="Wrong type:"+type_+"("+str(real_seq.id)+")")
    def _test_genome(self,real_genome,test_genome):
        self.assertEqual(set(real_genome.ANN_TYPES),set(test_genome.ANN_TYPES))
        self.assertEqual(len(real_genome.data),len(test_genome.data))
        for real_seq,test_seq in zip(real_genome.data,test_genome.data):
            self._test_seq(real_seq,test_seq)
    def test_one_plus_strand_all_utr_5(self):
        file_path = file_prefix + 'one_plus_strand_all_utr_5.tsv'
        real_genome = TestAnnGenomeCreator.real_genome.one_plus_strand_all_utr_5(False)  
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(RealGenome.genome_information,parser.result)
        creator.create()
        test_genome = creator.result
        self._test_genome(real_genome,test_genome)
    def test_two_plus_strand(self):
        file_path = file_prefix + 'two_plus_strand.tsv'
        real_genome = TestAnnGenomeCreator.real_genome.two_plus_strand(False)  
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(RealGenome.genome_information,parser.result)
        creator.create()
        test_genome = creator.result
        self._test_genome(real_genome,test_genome)
    def test_two_seq_on_same_plus(self):
        file_path = file_prefix + 'two_seq_on_same_plus_strand.tsv'
        real_genome = TestAnnGenomeCreator.real_genome.two_seq_on_same_plus_strand(False)  
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(RealGenome.genome_information,parser.result)
        creator.create()
        test_genome = creator.result
        self._test_genome(real_genome,test_genome)
    def test_one_plus_strand_both(self):
        file_path = file_prefix + 'one_plus_strand_both_utr.tsv'
        real_genome = TestAnnGenomeCreator.real_genome.one_plus_strand_both_utr(False)  
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(RealGenome.genome_information,parser.result)
        creator.create()
        test_genome = creator.result
        self._test_genome(real_genome,test_genome)
    def test_minus_strand(self):
        file_path = file_prefix + 'minus_strand.tsv'
        real_genome = TestAnnGenomeCreator.real_genome.minus_strand(False)  
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(RealGenome.genome_information,parser.result)
        creator.create()
        test_genome = creator.result
        self._test_genome(real_genome,test_genome)
    def test_multiple_utr_intron_cds(self):
        file_path = file_prefix + 'multiple_utr_intron_cds.tsv'
        real_genome = TestAnnGenomeCreator.real_genome.multiple_utr_intron_cds(False)  
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(RealGenome.genome_information,parser.result)
        creator.create()
        test_genome = creator.result
        self._test_genome(real_genome,test_genome)

if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestAnnGenomeCreator)
    unittest.main()
