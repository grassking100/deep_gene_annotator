import os
import shutil
import unittest
from sequence_annotation.process.performance import main
from sequence_annotation.utils.utils import read_json

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.join(ROOT, 'result')
DATA_ROOT = os.path.join(ROOT, 'data')


class TestPerformance(unittest.TestCase):
    def _test(self, name):
        saved_root = os.path.join(RESULT_ROOT, name)
        data_root = os.path.join(DATA_ROOT, name)
        predict_path = os.path.join(data_root, 'predict.gff3')
        answer_path = os.path.join(data_root, 'answer.gff3')
        region_table_path = os.path.join(data_root, 'region.tsv')
        main(predict_path,
             answer_path,
             region_table_path,
             saved_root,
             chrom_target='new_id')
        result = read_json(os.path.join(saved_root, 'block_performance.json'))
        expect_result = read_json(
            os.path.join(data_root, 'expect_part_block_performance.json'))
        for key, expect_val in expect_result.items():
            self.assertEqual(expect_val, result[key],
                             "Something is wrong at {}".format(key))

        main(answer_path,
             answer_path,
             region_table_path,
             saved_root,
             chrom_target='new_id')
        result = read_json(os.path.join(saved_root, 'block_performance.json'))
        expect_result = read_json(
            os.path.join(
                data_root,
                'expect_part_block_performance_for_full_correct.json'))
        for key, expect_val in expect_result.items():
            self.assertEqual(
                expect_val, result[key],
                "Something is wrong at {} of all correct test".format(key))

        shutil.rmtree(RESULT_ROOT)

    def test_single_transcript_single_exon(self):
        self._test('single_transcript_single_exon')

    def test_single_transcript_multiple_exon(self):
        self._test('single_transcript_multiple_exon')

    def test_multiple_transcript_multiple_exon(self):
        self._test('multiple_transcript_multiple_exon')

    def test_merged_gene(self):
        self._test('merged_gene')

    def test_single_transcript_single_intron(self):
        self._test('single_transcript_single_intron')

    def test_single_transcript_five_end_intron(self):
        self._test('single_transcript_five_end_intron')

    def test_single_transcript_three_end_intron(self):
        self._test('single_transcript_three_end_intron')
