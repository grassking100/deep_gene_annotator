from . import AnnSeqTestCase
import numpy as np
from . import ProcessedStatusNotSatisfied
from . import AnnSequence
from . import AnnSeqProcessor
class TestAnnSeqProcessor(AnnSeqTestCase):
    def test_normalized_all_types(self):
        ann = AnnSequence()
        ann.length = 8
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        real = AnnSequence().from_dict(ann.to_dict())
        real.init_space()
        ann.init_space()
        ann.set_ann('exon',1,0,1).set_ann('intron',1,1,3).set_ann('other',1,3,5)
        ann.set_ann('other',2,6,6).set_ann('exon',1,5,7)
        normalized = AnnSeqProcessor().get_normalized(ann)
        real.set_ann('exon',1,0,0).set_ann('exon',.5,1,1)
        real.set_ann('intron',.5,1,1).set_ann('intron',1,2,2).set_ann('intron',.5,3,3)
        real.set_ann('other',.5,3,3).set_ann('other',1,4,4)
        real.set_ann('other',.5,5,5).set_ann('other',2/3,6,6)
        real.set_ann('exon',.5,5,5).set_ann('exon',1/3,6,6).set_ann('exon',1,7,7)
        self.assert_seq_equal(real,normalized)
    def test_normalized_some_type(self):
        ann = AnnSequence()
        ann.length = 8
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        real = AnnSequence().from_dict(ann.to_dict())
        real.init_space()
        ann.init_space()
        ann.set_ann('exon',1,0,1).set_ann('intron',1,1,3).set_ann('other',1,3,5)
        ann.set_ann('other',2,6,6).set_ann('exon',1,5,7)
        #Make position 4 to intron
        ann.set_ann('intron',1,4,4)
        normalized = AnnSeqProcessor().get_normalized(ann,['exon','intron'])
        real.set_ann('exon',1,0,0).set_ann('exon',.5,1,1)
        real.set_ann('intron',.5,1,1).set_ann('intron',1,2,3)
        real.set_ann('other',1,3,5).set_ann('other',2,6,6)
        real.set_ann('exon',1,5,7).set_ann('intron',1,4,4)
        self.assert_seq_equal(real,normalized)
    def test_max_one_hot_all_types(self):
        ann = AnnSequence()
        ann.length = 8
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        real = AnnSequence().from_dict(ann.to_dict())
        real.init_space()
        ann.init_space()
        ann.set_ann('exon',1,0,1).set_ann('intron',1,1,3).set_ann('other',1,3,5)
        ann.set_ann('other',2,6,6).set_ann('exon',1,5,7)
        normalized = AnnSeqProcessor().get_one_hot(ann)
        real.set_ann('exon',1,0,1)
        real.set_ann('intron',1,2,3)
        real.set_ann('other',1,4,4).set_ann('exon',1,5,5)
        real.set_ann('other',1,6,6).set_ann('exon',1,7,7)
        self.assert_seq_equal(real,normalized)
    def test_max_one_hot_all_types_by_specific_order(self):
        ann = AnnSequence()
        ann.length = 8
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        real = AnnSequence().from_dict(ann.to_dict())
        real.init_space()
        ann.init_space()
        ann.set_ann('exon',1,0,1).set_ann('intron',1,1,3).set_ann('other',1,3,5)
        ann.set_ann('other',2,6,6).set_ann('exon',1,5,7)
        normalized = AnnSeqProcessor().get_one_hot(ann,['intron','exon','other'])
        real.set_ann('exon',1,0,0)
        real.set_ann('intron',1,1,3)
        real.set_ann('other',1,4,4).set_ann('exon',1,5,5)
        real.set_ann('other',1,6,6).set_ann('exon',1,7,7)
        self.assert_seq_equal(real,normalized)
    def test_max_one_hot_some_types_by_specific_order(self):
        ann = AnnSequence()
        ann.length = 8
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        real = AnnSequence().from_dict(ann.to_dict())
        real.init_space()
        ann.init_space()
        ann.set_ann('exon',1,0,1).set_ann('intron',1,1,3).set_ann('other',1,3,5)
        ann.set_ann('other',2,6,6).set_ann('exon',1,5,7)
        #Make position 4 to intron
        ann.set_ann('intron',1,4,4)
        normalized = AnnSeqProcessor().get_one_hot(ann,['intron','exon'])
        real.set_ann('exon',1,0,0).set_ann('intron',1,1,4)
        real.set_ann('other',1,3,5).set_ann('other',2,6,6)
        real.set_ann('exon',1,5,7)
        self.assert_seq_equal(real,normalized)
    def test_is_full_annotated_all_types(self):
        ann = AnnSequence()
        ann.length = 4
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        ann.init_space()
        ann.set_ann('intron',1,0,0).set_ann('exon',1,0,1).set_ann('other',1,2,3)
        status = AnnSeqProcessor().is_full_annotated(ann)
        self.assertTrue(status)
    def test_is_not_full_annotated_all_types(self):
        ann = AnnSequence()
        ann.length = 4
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        ann.init_space()
        ann.set_ann('intron',1,0,0).set_ann('exon',1,0,1).set_ann('other',1,2,2)
        status = AnnSeqProcessor().is_full_annotated(ann)
        self.assertFalse(status)
    def test_is_not_full_annotated_some_types(self):
        ann = AnnSequence()
        ann.length = 4
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        ann.init_space()
        ann.set_ann('intron',1,0,0).set_ann('exon',1,0,1).set_ann('other',1,2,3)
        status = AnnSeqProcessor().is_full_annotated(ann,['exon','intron'])
        self.assertFalse(status)
    def test_order_one_hot_all_types(self):
        ann = AnnSequence()
        ann.length = 8
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        real = AnnSequence().from_dict(ann.to_dict())
        real.init_space()
        ann.init_space()
        ann.set_ann('exon',1,0,1).set_ann('intron',1,1,3).set_ann('other',1,3,5)
        ann.set_ann('other',2,6,6).set_ann('exon',1,5,7)
        normalized = AnnSeqProcessor().get_one_hot(ann,method='order')
        real.set_ann('exon',1,0,1).set_ann('intron',1,2,3)
        real.set_ann('other',1,4,4).set_ann('exon',1,5,7)
        self.assert_seq_equal(real,normalized)
    def test_get_certain_status(self):
        ann = AnnSequence()
        ann.length = 9
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        ann.init_space()
        ann.set_ann('exon',1,0,2).set_ann('intron',1,3,5)
        ann.set_ann('intron',0.4,7,7).set_ann('exon',1,8,8)
        ann.processed_status='normalized'
        certain_status = AnnSeqProcessor().get_certain_status(ann)
        real_status = [True,True,True,True,True,True,False,False,True]
    def test_unnormalized_seq_certain_status(self):
        ann = AnnSequence()
        ann.length = 9
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        ann.init_space()
        ann.set_ann('exon',1,0,2).set_ann('intron',1,3,5)
        with self.assertRaises(ProcessedStatusNotSatisfied):
            certain_status = AnnSeqProcessor().get_certain_status(ann)
if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestAnnSeqProcessor)
    unittest.main()
