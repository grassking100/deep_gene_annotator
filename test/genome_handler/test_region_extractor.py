from ..seq_info_test_case import SeqInfoTestCase
from sequence_annotation.genome_handler.sequence import AnnSequence, SeqInformation
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
from sequence_annotation.genome_handler.region_extractor import RegionExtractor


class TestRegionExtractor(SeqInfoTestCase):
    def test_region_extract(self):
        # Create sequence to test
        chrom = AnnSequence()
        chrom.chromosome_id = 1
        chrom.strand = 'plus'
        chrom.length = 7
        chrom.id = 'region_1'
        chrom.ANN_TYPES = ['exon', 'intron', 'other', 'unsure', 'sure']
        answer = AnnSequence().from_dict(chrom.to_dict())
        answer.init_space()
        chrom.init_space()
        chrom.add_ann("exon", 1, 0,
                      1).add_ann("intron", 1, 2,
                                 4).add_ann("other", 1, 5,
                                            6).add_ann("sure", 1, 0, 6)
        extractor = RegionExtractor()
        regions = extractor.extract(chrom)
        seqinfos = SeqInfoContainer()
        ids = [
            'region_1_exon_1', 'region_1_intron_1', 'region_1_other_1',
            'region_1_sure_1'
        ]
        types = ['exon', 'intron', 'other', 'sure']
        for type_, id_, start, end in zip(types, ids, [0, 2, 5, 0],
                                          [1, 4, 6, 6]):
            seqinfo = SeqInformation()
            seqinfo.chromosome_id = 1
            seqinfo.strand = 'plus'
            seqinfo.ann_status = 'whole'
            seqinfo.ann_type = type_
            seqinfo.id = id_
            seqinfo.start = start
            seqinfo.end = end
            seqinfos.add(seqinfo)
        for region in regions:
            seqinfo = seqinfos.get(region.id)
            self.assert_seq_equal(seqinfo, region)
