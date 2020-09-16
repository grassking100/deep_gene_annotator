from ..ann_seq_test_case import AnnSeqTestCase
from sequence_annotation.utils.exception import ValueOutOfRange, NegativeNumberException
from sequence_annotation.file_process.utils import InvalidStrandType
from sequence_annotation.genome_handler.ann_seq_converter import CodingBedSeqConverter
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.exception import NotOneHotException, NotSameSizeException


class TestAnnSeqConverter(AnnSeqTestCase):
    def test_data(self):
        converter = CodingBedSeqConverter()
        data = {
            'strand': 'plus',
            'id': 'test',
            'chr': '1',
            'start': 10,
            'end': 300,
            'block_related_start': [0, 40, 90, 140, 190],
            'block_related_end': [30, 80, 130, 180, 290],
            'thick_start': 61,
            'thick_end': 160,
        }
        ann_seq = AnnSequence()
        ann_seq.length = 291
        ann_seq.id = 'test'
        ann_seq.chromosome_id = '1'
        ann_seq.strand = 'plus'
        ann_seq.ANN_TYPES = ['cds', 'intron', 'utr_5', 'utr_3']
        ann_seq.init_space()
        ann_seq.set_ann('utr_5', 1, 0, 30).set_ann('utr_5', 1, 40, 50)
        ann_seq.set_ann('cds', 1, 51, 80).set_ann('cds', 1, 90, 130)
        ann_seq.set_ann('cds', 1, 140, 150)
        ann_seq.set_ann('utr_3', 1, 151, 180).set_ann('utr_3', 1, 190, 290)
        ann_seq.set_ann('intron', 1, 31, 39).set_ann('intron', 1, 81, 89)
        ann_seq.set_ann('intron', 1, 131, 139).set_ann('intron', 1, 181, 189)
        result = converter.convert(data)
        self.assert_seq_equal(ann_seq, result)

    def test_data_is_not_one_hot(self):
        converter = CodingBedSeqConverter()
        data = {
            'strand': 'plus',
            'id': 'test',
            'chr': '1',
            'start': 10,
            'end': 300,
            'block_related_start': [0, 40, 90, 140, 190],
            'block_related_end': [30, 80, 130, 200, 290],
            'thick_start': 61,
            'thick_end': 160,
        }
        with self.assertRaises(NotOneHotException):
            converter.convert(data)

    def test_data_wrong_strand(self):
        converter = CodingBedSeqConverter()
        data = {
            'strand': 'x',
            'id': 'test',
            'chr': '1',
            'start': 10,
            'end': 300,
            'block_related_start': [0, 40, 90, 140, 190],
            'block_related_end': [30, 80, 130, 180, 290],
            'thick_start': 61,
            'thick_end': 160,
        }
        with self.assertRaises(InvalidStrandType):
            converter.convert(data)

    def test_data_wrong_num(self):
        converter = CodingBedSeqConverter()
        data = {
            'strand': 'plus',
            'id': 'test',
            'chr': '1',
            'start': 10,
            'end': 300,
            'block_related_start': [0, 40, 90, 140, 190],
            'block_related_end': [30, 80, 130, 180],
            'thick_start': 61,
            'thick_end': 160,
        }
        with self.assertRaises(NotSameSizeException):
            converter.convert(data)

    def test_data_wrong_thick(self):
        converter = CodingBedSeqConverter()
        data = {
            'strand': 'plus',
            'id': 'test',
            'chr': '1',
            'start': 10,
            'end': 300,
            'block_related_start': [0, 40, 90, 140, 190],
            'block_related_end': [30, 80, 130, 180, 290],
            'thick_start': 5,
            'thick_end': 160,
        }
        with self.assertRaises(NegativeNumberException):
            converter.convert(data)

    def test_data_out_range(self):
        converter = CodingBedSeqConverter()
        data = {
            'strand': 'plus',
            'id': 'test',
            'chr': '1',
            'start': 10,
            'end': 300,
            'block_related_start': [0, 40, 90, 140, 190],
            'block_related_end': [30, 80, 130, 180, 500],
            'thick_start': 61,
            'thick_end': 160,
        }
        with self.assertRaises(ValueOutOfRange):
            converter.convert(data)
