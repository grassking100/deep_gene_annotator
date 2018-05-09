import unittest
import numpy as np
from . import AnnSequence
from . import RegionExtractor
from . import SeqInfoGenerator
from . import AnnSeqProcessor
class TestSeqInfoGenerator(unittest.TestCase):
    principle = {'remove_end_of_strand':True,
                 'with_random_choose':True,
                 'replaceable':True,
                 "each_region_number":{'other':10,'cds':100},
                 'sample_number_per_region':6,
                 'half_length':4,
                 'max_diff':3,
                 'length_constant':False}
    chroms_info = {'chr1':120,'chr2':35}
    ANN_TYPES = ['cds','intron','utr_5','utr_3','other']
    frontground_types = ['cds','intron','utr_5','utr_3']
    background_type = 'other'
    source = "template"
    def _create_chrom(self,chrom_id,strand):
        chrom = AnnSequence()
        chrom.chromosome_id = chrom_id
        chrom.strand = strand
        chrom.length = TestSeqInfoGenerator.chroms_info[chrom_id]
        chrom.id = chrom_id+"_"+strand
        chrom.ANN_TYPES = TestSeqInfoGenerator.ANN_TYPES
        chrom.source = TestSeqInfoGenerator.source
        chrom.init_space()
        return chrom
    def _add_seq1(self,chrom):
        chrom.add_ann("utr_5",1,101,101).add_ann("cds",1,102,104)
        chrom.add_ann("intron",1,105,105)
        chrom.add_ann("cds",1,106,109).add_ann("utr_3",1,100,111)
        chrom.add_ann("utr_5",1,10,20)
    def test_total_number(self):
        #Create sequence to test
        chrom = self._create_chrom("chr1","plus")
        self._add_seq1(chrom)
        processor = AnnSeqProcessor()

        one_hot_chrom = processor.get_one_hot(chrom,
                                              TestSeqInfoGenerator.frontground_types,
                                              TestSeqInfoGenerator.background_type)
        extractor = RegionExtractor()
        regions = extractor.extract(one_hot_chrom)
        generator = SeqInfoGenerator()
        seeds,seqs_info = generator.generate(regions,
                                             TestSeqInfoGenerator.principle,
                                             TestSeqInfoGenerator.chroms_info,
                                             "seed","seq")
        seed_number = sum(TestSeqInfoGenerator.principle['each_region_number'].values())
        self.assertEqual(seed_number,len(seeds.data))
        seq_number = seed_number*TestSeqInfoGenerator.principle['sample_number_per_region']
        self.assertEqual(seq_number,len(seqs_info.data))        
if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestSeqInfoGenerator)
    unittest.main()
