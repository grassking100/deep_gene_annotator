import unittest
from sequence_annotation.process.data_processor import AnnSeqProcessor
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.sequence import AnnSequence


class TestDataContainer(unittest.TestCase):
    def test_ann_seq_data(self):
        try:
            CHANNEL_ORDER = ['exon', 'intron']
            ann_seqs = AnnSeqContainer()
            ann_seqs.ANN_TYPES = CHANNEL_ORDER
            ann_seq = AnnSequence()
            ann_seq.length = 5
            ann_seq.id = 'A'
            ann_seq.strand = 'plus'
            ann_seq.ANN_TYPES = CHANNEL_ORDER
            ann_seq.init_space()
            ann_seq.set_ann('exon', 1, 3, 4)
            ann_seq.set_ann('intron', 1, 0, 2)
            ann_seqs.add(ann_seq)
            ann_seq2 = AnnSequence()
            ann_seq2.length = 4
            ann_seq2.id = 'B'
            ann_seq2.strand = 'plus'
            ann_seq2.ANN_TYPES = CHANNEL_ORDER
            ann_seq2.init_space()
            ann_seq2.set_ann('exon', 1, 3, 3)
            ann_seq2.set_ann('intron', 1, 0, 2)
            ann_seqs.add(ann_seq2)
            data = {
                    'seq': {
                        'A': 'AATCG',
                        'B': 'TTTC'
                    },
                    'answer': ann_seqs
            }
            data_processor = AnnSeqProcessor(CHANNEL_ORDER)
            data_processor.process(data)
        except Exception:
            self.fail("There are some unexpected exception occur.")
