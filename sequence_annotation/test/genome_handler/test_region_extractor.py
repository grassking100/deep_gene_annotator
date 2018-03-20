import unittest
import numpy as np
from . import AnnSequence
from . import RegionExtractor
class TestRegionExtractor(unittest.TestCase):
    ANN_TYPES = ['cds','intron','utr_5','utr_3','intergenic_region',]
    frontground_types = ['cds','intron','utr_5','utr_3']
    background_type = 'intergenic_region'
    source = "template"
    data = {"chrom1":30}
    def _create_chrom(self,chrom_id,strand):
        chrom = AnnSequence()
        chrom.chromosome_id = chrom_id
        chrom.strand = strand
        chrom.length = TestRegionExtractor.data[chrom_id]
        chrom.id = chrom_id+"_"+strand
        chrom.ANN_TYPES = TestRegionExtractor.ANN_TYPES
        chrom.source = TestRegionExtractor.source
        chrom.initSpace()
        return chrom
    def _add_seq1(self,chrom):
        chrom.add_ann("utr_5",1,1,1).add_ann("cds",1,2,4).add_ann("intron",1,5,5)
        chrom.add_ann("cds",1,6,9).add_ann("utr_3",1,10,11)
    def _add_seq2(self,chrom):
        chrom.add_ann("utr_5",1,0,1).add_ann("cds",1,2,4).add_ann("intron",1,5,6)
        chrom.add_ann("cds",1,7,9).add_ann("utr_3",1,10,29)
    def test_seq2(self):
        #Create sequence to test
        chrom = self._create_chrom("chrom1","plus")
        self._add_seq2(chrom)
        extractor = RegionExtractor(chrom,TestRegionExtractor.frontground_types,TestRegionExtractor.background_type)
        extractor.extract()
        regions = extractor.result
        test = []
        for region in regions.data:
            test.append({'type':region.ann_type, 'start':region.start,
                         'end':region.end, 'strand':region.strand,
                         'chrom_id':region.chromosome_id})
        answers = [
            {"type":"cds","start":2,"end":4},
            {"type":"cds","start":7,"end":9},
            {"type":"intron","start":5,"end":6},
            {"type":"utr_5","start":0,"end":1},
            {"type":"utr_3","start":10,"end":29}
        ]
        for answer in answers:
            answer['strand'] = 'plus'
            answer['chrom_id'] = 'chrom1'
        self.assertEqual(answers,test)
    def test_seq1(self):
        #Create sequence to test
        chrom = self._create_chrom("chrom1","plus")
        self._add_seq1(chrom)
        extractor = RegionExtractor(chrom,TestRegionExtractor.frontground_types,TestRegionExtractor.background_type)
        extractor.extract()
        regions = extractor.result
        test = []
        for region in regions.data:
            test.append({'type':region.ann_type, 'start':region.start,
                         'end':region.end, 'strand':region.strand,
                         'chrom_id':region.chromosome_id})
        answers = [
            {"type":"cds","start":2,"end":4},
            {"type":"cds","start":6,"end":9},
            {"type":"intron","start":5,"end":5},
            {"type":"utr_5","start":1,"end":1},
            {"type":"utr_3","start":10,"end":11},
            {"type":"intergenic_region","start":0,"end":0},
            {"type":"intergenic_region","start":12,"end":29}
        ]
        for answer in answers:
            answer['strand'] = 'plus'
            answer['chrom_id'] = 'chrom1'
        self.assertEqual(answers,test)
if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestRegionExtractor)
    unittest.main()
