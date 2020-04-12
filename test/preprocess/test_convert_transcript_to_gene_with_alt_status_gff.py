import os
import unittest
from sequence_annotation.preprocess.convert_transcript_to_gene_with_alt_status_gff import main
from sequence_annotation.utils.utils import read_gff

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, 'data')


class TestAltStatusConverter(unittest.TestCase):
    def _test(self, name, select_site_by_election):
        data_root = os.path.join(DATA_ROOT, name)
        input_path = os.path.join(data_root, 'input.gff3')
        output_gff_path = os.path.join(data_root, 'output.gff3')
        answer_gff_path = os.path.join(data_root, 'answer.gff3')
        main(input_path,
             output_gff_path,
             select_site_by_election=select_site_by_election)
        output = read_gff(output_gff_path).sort_values(
            ['chr', 'strand', 'start', 'end',
             'feature']).reset_index(drop=True)
        answer = read_gff(answer_gff_path).sort_values(
            ['chr', 'strand', 'start', 'end',
             'feature']).reset_index(drop=True)

        self.assertTrue(answer.equals(output),
                        "Something wrong happen in {}".format(name))

    def test_multiple_start(self):
        self._test('multiple_start', True)

    def test_multiple_start_plus_strand(self):
        self._test('multiple_start_plus_strand', True)

    def test_multiple_end(self):
        self._test('multiple_end', True)

    def test_multiple_end_plus_strand(self):
        self._test('multiple_end_plus_strand', True)

    def test_multiple_start_end(self):
        self._test('multiple_start_end', True)

    def test_multiple_start_end_plus_strand(self):
        self._test('multiple_start_end_plus_strand', True)

    def test_multiple_equal_start(self):
        self._test('multiple_equal_start', True)

    def test_multiple_equal_end(self):
        self._test('multiple_equal_end', True)

    def test_multiple_exon(self):
        self._test('multiple_exon', True)

    def test_exon_skipping(self):
        self._test('exon_skipping', True)

    def test_intron_retention(self):
        self._test('intron_retention', True)

    def test_alt_acceptor(self):
        self._test('alt_acceptor', True)

    def test_alt_donor(self):
        self._test('alt_donor', True)

    def test_alt_donor_by_election(self):
        self._test('alt_donor_by_election', True)

    def test_alt_acceptor_by_election(self):
        self._test('alt_acceptor_by_election', True)

    def test_same_site_alt_acceptor_donor(self):
        self._test('same_site_alt_acceptor_donor', True)
