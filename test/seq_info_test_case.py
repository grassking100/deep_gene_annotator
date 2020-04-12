import unittest


class SeqInfoTestCase(unittest.TestCase):
    def assert_seq_equal(self, real_seq, test_seq):
        self.assertEqual(real_seq.chromosome_id, test_seq.chromosome_id)
        self.assertEqual(real_seq.id, test_seq.id)
        self.assertEqual(real_seq.strand, test_seq.strand)
        self.assertEqual(real_seq.length, test_seq.length)
        self.assertEqual(real_seq.source, test_seq.source)
        self.assertEqual(real_seq.start, test_seq.start)
        self.assertEqual(real_seq.end, test_seq.end)
        self.assertEqual(real_seq.end, test_seq.end)
        self.assertEqual(real_seq.extra_index, test_seq.extra_index)
        self.assertEqual(real_seq.extra_index_name, test_seq.extra_index_name)
        self.assertEqual(real_seq.ann_type, test_seq.ann_type)
        self.assertEqual(real_seq.ann_status, test_seq.ann_status)
