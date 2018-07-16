import unittest
import numpy as np
from . import SeqInformation
from . import SeqInfoContainer
from . import SeqInfoGenerator
class TestSeqInfoGenerator(unittest.TestCase):
    def test_total_number(self):
        principle = {'remove_end_of_strand':True,'with_random_choose':True,
                     'replaceable':True,
                     "each_region_number":{'intron':10,'exon':20},
                     "region_number_per_seed":6,'half_length':4,
                     'max_diff':3,'length_constant':False,"modes":['5\'']}
        chroms_info = {'chr1':100,'chr2':35}
        #Create sequence to test
        regions = SeqInfoContainer()
        for i in range(4):
            region = SeqInformation()
            region.chromosome_id = 'chr1'
            region.strand = 'plus'
            region.id = str(i)
            region.source = 'test'
            regions.add(region)
        region = regions.get('0')
        region.start = 0
        region.end = 20
        region.ann_type = 'intron'
        region = regions.get('1')
        region.start = 21
        region.end = 30
        region.ann_type = 'exon'
        region = regions.get('2')
        region.start = 31
        region.end = 50
        region.ann_type = 'intron'
        region = regions.get('3')
        region.start = 51
        region.end = 99
        region.ann_type = 'exon'
        generator = SeqInfoGenerator()
        seeds,seqs_info = generator.generate(regions,principle,chroms_info,"seed","seq")
        seed_number = sum(principle['each_region_number'].values())
        self.assertEqual(seed_number,len(seeds.data))
        seq_number = seed_number*principle['region_number_per_seed']
        self.assertEqual(seq_number,len(seqs_info.data))        
