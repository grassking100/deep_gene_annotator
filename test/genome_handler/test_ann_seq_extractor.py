import unittest
import numpy as np
from . import AnnSeqContainer
from . import SeqInfoContainer
from . import AnnSeqExtractor
from . import SeqInformation
from . import AnnSequence
class TestAnnSeqExtractor(unittest.TestCase):
    def _test_seq(self, real_seq, test_seq):
        self.assertEqual(real_seq.id, test_seq.id)
        self.assertEqual(real_seq.strand, test_seq.strand)
        self.assertEqual(real_seq.length, test_seq.length)
        self.assertEqual(real_seq.source, test_seq.source)
        for type_ in real_seq.ANN_TYPES:
            np.testing.assert_array_equal(real_seq.get_ann(type_),
                                          test_seq.get_ann(type_),
                                          err_msg="Wrong type:"+type_+"("+str(real_seq.id)+")")
    def test_extract_success(self):
        ann_seq_container = AnnSeqContainer()
        ann_seq_container.ANN_TYPES = ['exon','intron']
        """Create sequence to be extracted"""
        ann_seq = AnnSequence()
        ann_seq.ANN_TYPES = ['exon','intron']
        ann_seq.id = 'temp_plus'
        ann_seq.length = 10
        ann_seq.chromosome_id = 'temp'
        ann_seq.strand = 'plus'
        ann_seq.init_space()
        ann_seq.set_ann('exon',1,0,5).set_ann('intron',1,6,9)
        """Create real answer"""
        real_seq = AnnSequence()
        real_seq.ANN_TYPES = ['exon','intron']
        real_seq.id = 'extract'
        real_seq.length = 4
        real_seq.chromosome_id = 'temp'
        real_seq.strand = 'plus'
        real_seq.init_space()
        real_seq.set_ann('exon',1,0,0).set_ann('intron',1,1,3)
        """Create information to tell where to extract"""
        seq_info = SeqInformation()
        seq_info.chromosome_id = 'temp'
        seq_info.id = 'extract'
        seq_info.strand = 'plus'
        seq_info.start = 5
        seq_info.end = 8
        """Extract"""
        ann_seq_container.add(ann_seq)
        seq_info_container = SeqInfoContainer()
        seq_info_container.add(seq_info)
        extracted_seqs = AnnSeqExtractor().extract(ann_seq_container,seq_info_container)
        extracted_seq = extracted_seqs.get('extract')
        """Test"""
        self._test_seq(extracted_seq,real_seq)
if __name__=="__main__":    
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestAnnSeqExtractor)
    unittest.main()
