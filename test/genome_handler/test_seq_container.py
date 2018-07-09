import unittest
import numpy as np
import pandas as pd
from . import SeqInfoContainer
from . import SeqInformation
class TestSequenceContainer(unittest.TestCase):
    def test_iter_id(self):
        container = SeqInfoContainer()
        ids = list(range(10,2,-1))
        for i in ids:
            seq_info = SeqInformation()
            seq_info.id=i
            container.add(seq_info)
        iter_ids = []
        for seq_info in container:
            iter_ids.append(seq_info.id)
        self.assertEqual(sorted(ids),iter_ids)