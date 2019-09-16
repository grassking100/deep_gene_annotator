from . import AnnSeqTestCase
import numpy as np
from numpy.testing import assert_array_equal
from sequence_annotation.genome_handler.exception import ProcessedStatusNotSatisfied
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.ann_seq_processor import get_normalized,get_one_hot
from sequence_annotation.genome_handler.ann_seq_processor import is_full_annotated,get_certain_status,is_one_hot
from sequence_annotation.genome_handler.ann_genome_processor import genome2dict_vec
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer

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
        normalized = get_normalized(ann)
        real.processed_status = 'normalized'
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
        normalized = get_normalized(ann,['exon','intron'])
        real.set_ann('exon',1,0,0).set_ann('exon',.5,1,1)
        real.set_ann('intron',.5,1,1).set_ann('intron',1,2,3)
        real.set_ann('other',1,3,5).set_ann('other',2,6,6)
        real.set_ann('exon',1,5,7).set_ann('intron',1,4,4)
        real.processed_status = 'normalized'
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
        normalized = get_one_hot(ann)
        real.set_ann('exon',1,0,1)
        real.set_ann('intron',1,2,3)
        real.set_ann('other',1,4,4).set_ann('exon',1,5,5)
        real.set_ann('other',1,6,6).set_ann('exon',1,7,7)
        real.processed_status = 'one_hot'
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
        normalized = get_one_hot(ann,['intron','exon','other'])
        real.set_ann('exon',1,0,0)
        real.set_ann('intron',1,1,3)
        real.set_ann('other',1,4,4).set_ann('exon',1,5,5)
        real.set_ann('other',1,6,6).set_ann('exon',1,7,7)
        real.processed_status = 'one_hot'
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
        normalized = get_one_hot(ann,['intron','exon'])
        real.set_ann('exon',1,0,0).set_ann('intron',1,1,4)
        real.set_ann('other',1,3,5).set_ann('other',2,6,6)
        real.set_ann('exon',1,5,7)
        real.processed_status = 'one_hot'
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
        status = is_full_annotated(ann)
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
        status = is_full_annotated(ann)
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
        status = is_full_annotated(ann,['exon','intron'])
        self.assertFalse(status)

    def test_is_one_hot(self):
        ann = AnnSequence()
        ann.length = 4
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        ann.init_space()
        ann.set_ann('intron',1,0,0).set_ann('exon',1,1,3).set_ann('other',1,2,3)
        status = is_one_hot(ann,['exon','intron'])
        self.assertTrue(status)

    def test_is_not_one_hot(self):
        ann = AnnSequence()
        ann.length = 4
        ann.strand='plus'
        ann.chromosome_id='1'
        ann.id=1
        ann.ANN_TYPES = ['exon','intron','other']
        ann.init_space()
        ann.set_ann('intron',1,0,0).set_ann('exon',1,0,1).set_ann('other',1,2,3)
        status = is_one_hot(ann,['exon','intron'])
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
        normalized = get_one_hot(ann,method='order')
        real.set_ann('exon',1,0,1).set_ann('intron',1,2,3)
        real.set_ann('other',1,4,4).set_ann('exon',1,5,7)
        real.processed_status = 'one_hot'
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
        certain_statuses = get_certain_status(ann)
        real_statuses = [True,True,True,True,True,True,False,True,True]
        for certain_status,real_status in zip(certain_statuses,real_statuses):
            self.assertEqual(real_status,certain_status)

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
            certain_status = get_certain_status(ann)

    def test_get_ann_vecs(self):
        genome = AnnSeqContainer()
        genome.ANN_TYPES = ['gene','other']
        seq1 = AnnSequence()
        seq1.id = 'seq1'
        seq1.strand = 'plus'
        seq1.chromosome_id = 'test'
        seq1.ANN_TYPES = ['gene','other']
        seq1.length = 10
        seq1.init_space()
        seq1.set_ann('gene',1,1,3).set_ann('other',1,4,9)
        seq1.set_ann('other',1,0,0)
        seq2 = AnnSequence()
        seq2.id = 'seq2'
        seq2.strand = 'minus'
        seq2.chromosome_id = 'test'
        seq2.ANN_TYPES = ['gene','other']
        seq2.length = 10
        seq2.init_space()
        seq2.set_ann('gene',1,1,3)
        seq2.set_ann('other',1,4,9)
        seq2.set_ann('other',1,0,0)
        genome.add(seq1)
        genome.add(seq2)
        ann_vecs = genome2dict_vec(genome,['gene','other'])
        answer =   {'seq1':np.transpose(np.array([[0,1,1,1,0,0,0,0,0,0],[1,0,0,0,1,1,1,1,1,1]])),
                    'seq2':np.transpose(np.array([[0,0,0,0,0,0,1,1,1,0],[1,1,1,1,1,1,0,0,0,1]]))
                   }
        assert_array_equal(answer['seq1'],ann_vecs['seq1'])
        assert_array_equal(answer['seq2'],ann_vecs['seq2'])
