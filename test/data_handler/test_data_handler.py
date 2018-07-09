from . import AnnSeqContainer,AnnSequence
from . import SeqAnnDataHandler
import unittest
import numpy as np
from numpy.testing import assert_array_equal

class TestDataHandler(unittest.TestCase):
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
        ann_vecs = SeqAnnDataHandler.get_ann_vecs(genome,['gene','other'])
        answer =   {'seq1':np.transpose(np.array([[0,1,1,1,0,0,0,0,0,0],[1,0,0,0,1,1,1,1,1,1]])),
                    'seq2':np.transpose(np.array([[0,0,0,0,0,0,1,1,1,0],[1,1,1,1,1,1,0,0,0,1]]))
                   }
        assert_array_equal(answer['seq1'],ann_vecs['seq1'])
        assert_array_equal(answer['seq2'],ann_vecs['seq2'])
