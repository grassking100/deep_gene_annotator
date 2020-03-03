import unittest
from sequence_annotation.process.utils import get_seq_mask

class TestUtils(unittest.TestCase):
    def test_get_seq_mask(self):
        masks = get_seq_mask([3,2,1])
        masks = masks.cpu().numpy().tolist()
        self.assertEqual([[1,1,1],
                          [1,1,0],
                          [1,0,0]], masks)
