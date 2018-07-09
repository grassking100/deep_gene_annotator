from . import SeqInfoTestCase
import numpy as np
from . import AnnSequence
from . import SeqInformation
from . import SeqInfoContainer
from . import RegionExtractor
class TestRegionExtractor(SeqInfoTestCase):
    def test_region_extract(self):
        #Create sequence to test
        chrom = AnnSequence()
        chrom.chromosome_id = '1'
        chrom.strand = 'plus'
        chrom.length = 7
        chrom.id = 1
        chrom.ANN_TYPES = ['exon','intron','other']
        answer = AnnSequence().from_dict(chrom.to_dict())
        answer.init_space()
        chrom.init_space()
        chrom.add_ann("exon",1,0,1).add_ann("intron",1,2,4).add_ann("other",1,5,6)
        chrom.processed_status = 'one_hot'
        extractor = RegionExtractor()
        regions = extractor.extract(chrom)
        seqinfos = SeqInfoContainer()
        for i in range(0,3):
            seqinfo = SeqInformation()
            seqinfo.chromosome_id = '1'
            seqinfo.strand = 'plus'
            seqinfo.start = 0
            seqinfo.end = 4
            seqinfo.id = 'region_1_plus_'+str(i)
            seqinfos.add(seqinfo)
        seq = seqinfos.get('region_1_plus_0')
        seq.start = 0
        seq.end = 1
        seq.ann_type = 'exon'
        seq.ann_status = 'whole'
        seq = seqinfos.get('region_1_plus_1')
        seq.start = 2
        seq.end = 4
        seq.ann_type = 'intron'
        seq.ann_status = 'whole'
        seq = seqinfos.get('region_1_plus_2')
        seq.start = 5
        seq.end = 6
        seq.ann_type = 'other'
        seq.ann_status = 'whole'
        for region in regions:
            seqinfo = seqinfos.get(region.id)
            self.assert_seq_equal(seqinfo,region)
