import unittest
import numpy as np
import os
import sys
sys.path.append((os.path.abspath(__file__+"/../..")))
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.region_extractor import RegionExtractor
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator
class TestSeqInfoGenerator(unittest.TestCase):
    principle = {'remove_end_of_strand':True,
                 'with_random_choose':True,
                 'replaceable':True,
                 "each_region_number":{'intergenic_region':10,'cds':100},
                 'sample_number_per_region':6,
                 'half_length':4,
                  'max_diff':3}
    chroms_info = {'chr1':120,'chr2':35}
    ANN_TYPES = ['cds','intron','utr_5','utr_3','intergenic_region']
    frontground_types = ['cds','intron','utr_5','utr_3']
    background_type = 'intergenic_region'
    source = "template"
    def _create_chrom(self,chrom_id,strand):
        chrom = AnnSequence()
        chrom.chromosome_id = chrom_id
        chrom.strand = strand
        chrom.length = TestSeqInfoGenerator.chroms_info[chrom_id]
        chrom.id = chrom_id+"_"+strand
        chrom.ANN_TYPES = TestSeqInfoGenerator.ANN_TYPES
        chrom.source = TestSeqInfoGenerator.source
        chrom.initSpace()
        return chrom
    def _add_seq1(self,chrom):
        chrom.add_ann("utr_5",1,101,101).add_ann("cds",1,102,104)
        chrom.add_ann("intron",1,105,105)
        chrom.add_ann("cds",1,106,109).add_ann("utr_3",1,100,111)
        chrom.add_ann("utr_5",1,10,20)
    def test_number(self):
        #Create sequence to test
        chrom = self._create_chrom("chr1","plus")
        self._add_seq1(chrom)
        extractor = RegionExtractor(chrom,
                                    TestSeqInfoGenerator.frontground_types,
                                    TestSeqInfoGenerator.background_type)
        extractor.extract()
        regions = extractor.result
        generator = SeqInfoGenerator(regions,
                                     TestSeqInfoGenerator.principle,
                                     TestSeqInfoGenerator.chroms_info,
                                     "seed","seq")
        generator.generate()
        seed_number = sum(TestSeqInfoGenerator.principle['each_region_number'].values())
        self.assertEqual(seed_number,len(generator.seeds.data))
        seq_number = seed_number*TestSeqInfoGenerator.principle['sample_number_per_region']
        self.assertEqual(seq_number,len(generator.seqs_info.data))
        
if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestSeqInfoGenerator)
    unittest.main()
