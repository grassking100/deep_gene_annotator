import unittest
import numpy as np
from sequence_annotation.utils.exception import ReturnNoneException
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.seq_info_parser import USCUParser
from sequence_annotation.genome_handler.ann_genome_creator import AnnGenomeCreator
from . import RealGenome
class TestAnnGenomeCreator(unittest.TestCase):
    genome_information={'chromosome':{'chr1':30,'chr2':35},'source':'unit_test'}
    real_genome = RealGenome()
    def test_not_create(self):
        file_path = 'data/ucsc/one_plus_strand_all_utr_5.tsv'
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(TestAnnGenomeCreator.genome_information,parser.data)
        with self.assertRaises(ReturnNoneException):
            data = creator.data
    def test_create_type(self):
        file_path = 'data/ucsc/one_plus_strand_all_utr_5.tsv'
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(TestAnnGenomeCreator.genome_information,parser.data)
        creator.create()
        self.assertEqual(AnnSeqContainer,type(creator.data))
    def _test_equal(self,real_genome,test_genome):
        self.assertEqual(set(real_genome.ANN_TYPES),set(test_genome.ANN_TYPES))
        self.assertEqual(len(real_genome.data),len(test_genome.data))
        for real_seq,test_seq in zip(real_genome.data,test_genome.data):
            self.assertEqual(real_seq.id,test_seq.id)
            self.assertEqual(real_seq.strand,test_seq.strand)
            self.assertEqual(real_seq.length,test_seq.length)
            self.assertEqual(real_seq.source,test_seq.source)
            for type_ in real_seq.ANN_TYPES:
                np.testing.assert_array_equal(real_seq.get_ann(type_),
                                              test_seq.get_ann(type_),
                                              err_msg="Wrong type:"+type_+"("+str(real_seq.id)+")")
    def test_one_plus_strand_all_utr_5_without_intergenic_region(self):
        file_path = 'data/ucsc/one_plus_strand_all_utr_5.tsv'
        real_genome = TestAnnGenomeCreator.real_genome.one_plus_strand_all_utr_5(False)  
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(TestAnnGenomeCreator.genome_information,parser.data)
        creator.create()
        test_genome = creator.data
        self._test_equal(real_genome,test_genome)
    def test_two_plus_strand_without_intergenic_region(self):
        file_path = 'data/ucsc/two_plus_strand.tsv'
        real_genome = TestAnnGenomeCreator.real_genome.two_plus_strand(False)  
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(TestAnnGenomeCreator.genome_information,parser.data)
        creator.create()
        test_genome = creator.data
        self._test_equal(real_genome,test_genome)
    def test_two_seq_on_same_plus_strand_without_intergenic_region(self):
        file_path = 'data/ucsc/two_seq_on_same_plus_strand.tsv'
        real_genome = TestAnnGenomeCreator.real_genome.two_seq_on_same_plus_strand(False)  
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(TestAnnGenomeCreator.genome_information,parser.data)
        creator.create()
        test_genome = creator.data
        self._test_equal(real_genome,test_genome)
    def test_one_plus_strand_both_utr_without_intergenic_region(self):
        file_path = 'data/ucsc/one_plus_strand_both_utr.tsv'
        real_genome = TestAnnGenomeCreator.real_genome.one_plus_strand_both_utr(False)  
        parser = USCUParser(file_path)
        parser.parse()
        creator = AnnGenomeCreator(TestAnnGenomeCreator.genome_information,parser.data)
        creator.create()
        test_genome = creator.data
        self._test_equal(real_genome,test_genome)
if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestAnnGenomeCreator)
    unittest.main()

