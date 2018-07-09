import unittest
import numpy as np
class AnnSeqTestCase(unittest.TestCase):
    def assert_seq_equal(self, real_seq, test_seq):
        self.assertEqual(real_seq.chromosome_id, test_seq.chromosome_id)
        self.assertEqual(real_seq.id, test_seq.id)
        self.assertEqual(real_seq.strand, test_seq.strand)
        self.assertEqual(real_seq.length, test_seq.length)
        self.assertEqual(real_seq.source, test_seq.source)
        self.assertEqual(real_seq.processed_status, test_seq.processed_status)
        self.assertEqual(set(real_seq.ANN_TYPES), set(test_seq.ANN_TYPES))
        for type_ in real_seq.ANN_TYPES:
            err_msg="Wrong type:"+type_+"("+str(real_seq.id)+")"
            real_ann = real_seq.get_ann(type_)
            test_ann = test_seq.get_ann(type_)
            np.testing.assert_array_equal(real_ann,test_ann,err_msg=err_msg)