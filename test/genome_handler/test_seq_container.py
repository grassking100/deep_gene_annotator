import unittest
from sequence_annotation.genome_handler.seq_container import SeqInfoContainer
from sequence_annotation.genome_handler.sequence import SeqInformation


class TestSequenceContainer(unittest.TestCase):
    def test_iter_id(self):
        container = SeqInfoContainer()
        ids = list(range(10, 2, -1))
        for i in ids:
            seq_info = SeqInformation()
            seq_info.id = i
            container.add(seq_info)
        iter_ids = []
        for seq_info in container:
            iter_ids.append(seq_info.id)
        self.assertEqual(sorted(ids), iter_ids)
