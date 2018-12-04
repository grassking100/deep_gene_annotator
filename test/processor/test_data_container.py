import unittest
from sequence_annotation.processor.data_processor import SimpleData,AnnSeqData
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.utils.exception import DimensionNotSatisfy
class TestDataContainer(unittest.TestCase):
    def test_simple_data(self):
        try: 
            data = SimpleData({'training':{'inputs':[[1,0]],'answers':[[1,0]]}})
            data.before_process()
            data.process()
            data.after_process()
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
    def test_ann_seq_data(self):
        try:
            ann_seqs = AnnSeqContainer()
            ann_seqs.ANN_TYPES=['Black','White']
            ann_seq = AnnSequence()
            ann_seq.length = 5
            ann_seq.id='A'
            ann_seq.strand = 'plus'
            ann_seq.ANN_TYPES = ['Black','White']
            ann_seq.init_space()
            ann_seq.set_ann('Black',1,3,4)
            ann_seq.set_ann('White',1,0,2)
            ann_seqs.add(ann_seq)
            ann_seq2 = AnnSequence()
            ann_seq2.length = 4
            ann_seq2.id='B'
            ann_seq2.strand = 'plus'
            ann_seq2.ANN_TYPES = ['Black','White']
            ann_seq2.init_space()
            ann_seq2.set_ann('Black',1,3,3)
            ann_seq2.set_ann('White',1,0,2)
            ann_seqs.add(ann_seq2)
            data = AnnSeqData({'data':{'training':{'inputs':{'A':'AATCG','B':'TTTC'},
                                                   'answers':ann_seqs}},
                              'ANN_TYPES':['Black','White']},padding_value=-1)
            data.before_process()
            data.process()
            data.after_process()
        except Exception as e:
            raise e
            self.fail("There are some unexpected exception occur.")
    def test_ann_seq_unequal_length(self):
        ann_seqs = AnnSeqContainer()
        ann_seqs.ANN_TYPES=['Black','White']
        ann_seq = AnnSequence()
        ann_seq.length = 5
        ann_seq.id='A'
        ann_seq.strand = 'plus'
        ann_seq.ANN_TYPES = ['Black','White']
        ann_seq.init_space()
        ann_seq.set_ann('Black',1,3,4)
        ann_seq.set_ann('White',1,0,2)
        ann_seqs.add(ann_seq)
        ann_seq2 = AnnSequence()
        ann_seq2.length = 4
        ann_seq2.id='B'
        ann_seq2.strand = 'plus'
        ann_seq2.ANN_TYPES = ['Black','White']
        ann_seq2.init_space()
        ann_seq2.set_ann('Black',1,3,3)
        ann_seq2.set_ann('White',1,0,2)
        ann_seqs.add(ann_seq2)
        data = AnnSeqData({'data':{'training':{'inputs':{'A':'AATCG','B':'TTTC'},
                                               'answers':ann_seqs}},
                           'ANN_TYPES':['Black','White']},padding_value=None)
        data.before_process()
        self.assertRaises(DimensionNotSatisfy,data.process)
        data.after_process()