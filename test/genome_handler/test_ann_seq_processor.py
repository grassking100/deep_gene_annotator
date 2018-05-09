import unittest
import numpy as np
from . import AnnSequence
from . import AnnSeqProcessor
class TestAnnSeqProcessor(unittest.TestCase):
    ANN_TYPES = ['cds','intron','utr_5','utr_3','intergenic_region',]
    frontground_types = ['cds','intron','utr_5','utr_3']
    background_type = 'intergenic_region'
    source = "template"
    data = {"chrom1":30}
    def _create_chrom(self,chrom_id,strand):
        chrom = AnnSequence()
        chrom.chromosome_id = chrom_id
        chrom.strand = strand
        chrom.length = TestAnnSeqProcessor.data[chrom_id]
        chrom.id = chrom_id+"_"+strand
        chrom.ANN_TYPES = TestAnnSeqProcessor.ANN_TYPES
        chrom.source = TestAnnSeqProcessor.source
        chrom.init_space()
        return chrom
    def _test_seq(self, real_seq, test_seq):
        self.assertEqual(real_seq.id, test_seq.id)
        self.assertEqual(real_seq.strand, test_seq.strand)
        self.assertEqual(real_seq.length, test_seq.length)
        self.assertEqual(real_seq.source, test_seq.source)
        for type_ in real_seq.ANN_TYPES:
            np.testing.assert_array_equal(real_seq.get_ann(type_),
                                          test_seq.get_ann(type_),
                                          err_msg="Wrong type:"+type_+"("+str(real_seq.id)+")")
    def _add_seq1(self,chrom):
        chrom.add_ann("utr_5",1,1,1).add_ann("cds",1,2,4).add_ann("intron",1,5,5)
        chrom.add_ann("cds",1,6,9).add_ann("utr_3",1,10,11)
    def _add_seq2(self,chrom):
        chrom.add_ann("utr_5",1,1,1).add_ann("cds",1,2,4).add_ann("intron",1,5,6)
        chrom.add_ann("cds",1,7,9).add_ann("utr_3",1,10,14)
    def _add_seq3(self,chrom):
        chrom.add_ann("utr_5",1,17,20)
    def _add_seq4(self,chrom):
        chrom.add_ann("utr_5",1,1,2).add_ann("cds",1,3,4).add_ann("intron",1,5,5)
        chrom.add_ann("cds",1,6,9).add_ann("utr_3",1,10,11)
    def test_three_seq_normalized(self):
        """Three sequence are overlapped"""
        #Create sequence to test
        chrom = self._create_chrom("chrom1","plus")
        self._add_seq1(chrom)
        self._add_seq2(chrom)
        self._add_seq4(chrom)
        processor = AnnSeqProcessor()
        norm_chrom = processor.get_normalized(chrom,
                                              TestAnnSeqProcessor.frontground_types,
                                              TestAnnSeqProcessor.background_type)
        #Create answer
        answer = self._create_chrom("chrom1","plus")
        answer.add_ann("utr_5",1,1,1).add_ann("utr_5",1/3,2,2)
        answer.add_ann("cds",2/3,2,2).add_ann("cds",1,3,4)
        answer.add_ann("intron",1,5,5).add_ann("intron",1/3,6,6)
        answer.add_ann("cds",2/3,6,6).add_ann("cds",1,7,9)
        answer.add_ann("utr_3",1,10,14)
        answer.add_ann("intergenic_region",1,0,0)
        answer.add_ann("intergenic_region",1,15,29)
        #Test equality
        self._test_seq(answer,norm_chrom)

    def test_two_plus_one_seq_normalized(self):
        """Two sequence are overlapped and one is not"""
        #Create sequence to test
        chrom = self._create_chrom("chrom1","plus")
        self._add_seq1(chrom)
        self._add_seq2(chrom)
        self._add_seq3(chrom)
        processor = AnnSeqProcessor()
        norm_chrom = processor.get_normalized(chrom,
                                              TestAnnSeqProcessor.frontground_types,
                                              TestAnnSeqProcessor.background_type)
        #Create answer
        answer = self._create_chrom("chrom1","plus")
        answer.add_ann("utr_5",1,1,1).add_ann("cds",1,2,4).add_ann("intron",1,5,5)
        answer.add_ann("cds",1,7,9).add_ann("utr_3",1,10,14)
        answer.add_ann("intron",0.5,6,6).add_ann("cds",0.5,6,6)
        answer.add_ann("utr_5",1,17,20)
        answer.add_ann("intergenic_region",1,0,0)
        answer.add_ann("intergenic_region",1,15,16)
        answer.add_ann("intergenic_region",1,21,29)
        #Test equality
        self._test_seq(answer,norm_chrom)
    def test_two_seq_normalized(self):
        #Create sequence to test
        chrom = self._create_chrom("chrom1","plus")
        self._add_seq1(chrom)
        self._add_seq2(chrom)
        processor = AnnSeqProcessor()
        norm_chrom = processor.get_normalized(chrom,
                                              TestAnnSeqProcessor.frontground_types,
                                              TestAnnSeqProcessor.background_type)
        #Create answer
        answer = self._create_chrom("chrom1","plus")
        answer.add_ann("utr_5",1,1,1).add_ann("cds",1,2,4).add_ann("intron",1,5,5)
        answer.add_ann("intron",0.5,6,6).add_ann("cds",0.5,6,6)
        answer.add_ann("cds",1,7,9).add_ann("utr_3",1,10,14)
        answer.add_ann("intergenic_region",1,0,0)
        answer.add_ann("intergenic_region",1,15,29)
        #Test equality
        self._test_seq(answer,norm_chrom)
    def test_one_seq_normalized(self):
        #Create answer
        answer = self._create_chrom("chrom1","plus")
        self._add_seq1(answer)
        answer.add_ann("intergenic_region",1,0,0)
        answer.add_ann("intergenic_region",1,12,29)
        #Create sequence to test
        chrom = self._create_chrom("chrom1","plus")
        self._add_seq1(chrom)
        processor = AnnSeqProcessor()
        norm_chrom = processor.get_normalized(chrom,
                                              TestAnnSeqProcessor.frontground_types,
                                              TestAnnSeqProcessor.background_type)
        #Test equality
        self._test_seq(answer,norm_chrom)
    def test_two_plus_one_seq_one_hot(self):
        #Create sequence to test
        chrom = self._create_chrom("chrom1","plus")
        self._add_seq1(chrom)
        self._add_seq2(chrom)
        self._add_seq3(chrom)
        processor = AnnSeqProcessor()
        one_hot_chrom = processor.get_one_hot(chrom,
                                              TestAnnSeqProcessor.frontground_types,
                                              TestAnnSeqProcessor.background_type)
        #Create answer
        answer = self._create_chrom("chrom1","plus")
        answer.add_ann("utr_5",1,1,1).add_ann("cds",1,2,4).add_ann("intron",1,5,5)
        answer.add_ann("cds",1,7,9).add_ann("utr_3",1,10,14)
        answer.add_ann("cds",1,6,6)
        answer.add_ann("utr_5",1,17,20)
        answer.add_ann("intergenic_region",1,0,0)
        answer.add_ann("intergenic_region",1,15,16)
        answer.add_ann("intergenic_region",1,21,29)
        #Test equality
        self._test_seq(answer,one_hot_chrom)
    def test_two_seq_one_hot(self):
        #Create sequence to test
        chrom = self._create_chrom("chrom1","plus")
        self._add_seq1(chrom)
        self._add_seq2(chrom)
        processor = AnnSeqProcessor()
        one_hot_chrom = processor.get_one_hot(chrom,
                                              TestAnnSeqProcessor.frontground_types,
                                              TestAnnSeqProcessor.background_type)
        #Create answer
        answer = self._create_chrom("chrom1","plus")
        answer.add_ann("utr_5",1,1,1).add_ann("cds",1,2,4).add_ann("intron",1,5,5)
        answer.add_ann("cds",1,6,6)
        answer.add_ann("cds",1,7,9).add_ann("utr_3",1,10,14)
        answer.add_ann("intergenic_region",1,0,0)
        answer.add_ann("intergenic_region",1,15,29)
        #Test equality
        self._test_seq(answer,one_hot_chrom)
    def test_one_seq_one_hot(self):
        #Create answer
        answer = self._create_chrom("chrom1","plus")
        self._add_seq1(answer)
        answer.add_ann("intergenic_region",1,0,0)
        answer.add_ann("intergenic_region",1,12,29)
        #Create sequence to test
        chrom = self._create_chrom("chrom1","plus")
        self._add_seq1(chrom)
        processor = AnnSeqProcessor()
        one_hot_chrom = processor.get_one_hot(chrom,
                                              TestAnnSeqProcessor.frontground_types,
                                              TestAnnSeqProcessor.background_type)
        #Test equality
        self._test_seq(answer,one_hot_chrom)
    def test_three_seq_one_hot(self):
        """Three sequence are overlapped"""
        #Create sequence to test
        chrom = self._create_chrom("chrom1","plus")
        self._add_seq1(chrom)
        self._add_seq2(chrom)
        self._add_seq4(chrom)
        processor = AnnSeqProcessor()
        one_hot_chrom = processor.get_one_hot(chrom,
                                              TestAnnSeqProcessor.frontground_types,
                                              TestAnnSeqProcessor.background_type)
        #Create answer
        answer = self._create_chrom("chrom1","plus")
        answer.add_ann("utr_5",1,1,1)
        answer.add_ann("cds",1,2,2).add_ann("cds",1,3,4)
        answer.add_ann("intron",1,5,5)
        answer.add_ann("cds",1,6,6).add_ann("cds",1,7,9)
        answer.add_ann("utr_3",1,10,14)
        answer.add_ann("intergenic_region",1,0,0)
        answer.add_ann("intergenic_region",1,15,29)
        #Test equality
        self._test_seq(answer,one_hot_chrom)
if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestAnnSeqProcessor)
    unittest.main()
