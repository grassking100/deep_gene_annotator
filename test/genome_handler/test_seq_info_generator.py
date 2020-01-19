import unittest
from sequence_annotation.genome_handler.sequence import SeqInformation
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
from sequence_annotation.genome_handler.seq_info_generator import SeqInfoGenerator

class TestSeqInfoGenerator(unittest.TestCase):
    def test_total_number(self):
        chroms_info = {'chr1':100,'chr2':35}
        #Create sequence to test
        regions = SeqInfoContainer()
        for i in range(4):
            region = SeqInformation()
            region.chromosome_id = 'chr1'
            region.strand = 'plus'
            region.id = str(i)
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
        seeds,seqs_info = generator.generate(regions,chroms_info,half_length=4,modes=['five_prime','middle'])
        self.assertEqual(4,len(seeds.data))
        self.assertEqual(4,len(seqs_info.data))        
