import os
import math
import shutil
import unittest
from sequence_annotation.utils.utils import read_json
from sequence_annotation.process.performance import main

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.join(ROOT, 'result')
DATA_ROOT = os.path.join(ROOT, 'data')

def assert_json(test_case,name,answer,predict):
    if isinstance(answer,dict):
        for key, answer_val in answer.items():
            assert_json(test_case,key,answer_val,predict[key])
    elif isinstance(answer,list):
        test_case.assertEqual(len(answer),len(predict))
        for answer_val,predict_val in zip(answer,predict):
            assert_json(test_case,name,answer_val,predict_val)
    else:
        if math.isnan(answer):
            answer = 'nan'
        if math.isnan(predict):
            predict = 'nan'
        test_case.assertEqual(answer, predict,"Get {} but expect {} in {}".format(predict,answer,name)) 

def assert_equal(test_case,answer_root,predicted_root,full_correct=False,calculate_base=True):
    if full_correct:
        postfix='_for_full_correct'
    else:
        postfix=''
    names = ['site_matched','distance','block_performance']
    if calculate_base:
        names.append('base_performance')
    for name in names:
        answer_path = os.path.join(answer_root, '{}{}.json'.format(name,postfix))
        predict_path = os.path.join(predicted_root, '{}.json'.format(name))
        answer = read_json(answer_path)
        predict = read_json(predict_path)
        assert_json(test_case,name,answer,predict)

class TestPerformance(unittest.TestCase):
    def _test(self, name,calculate_base=True):
        saved_root = os.path.join(RESULT_ROOT, name)
        answer_root = os.path.join(DATA_ROOT, name)
        predict_path = os.path.join(answer_root, 'predict.gff3')
        answer_path = os.path.join(answer_root, 'answer.gff3')
        region_table_path = os.path.join(answer_root, 'region.tsv')
        main(predict_path,answer_path,region_table_path,saved_root,calculate_base=calculate_base)
        assert_equal(self,answer_root,saved_root,False,calculate_base)
        main(answer_path,answer_path,region_table_path,saved_root,calculate_base=calculate_base)
        assert_equal(self,answer_root,saved_root,True,calculate_base)
        shutil.rmtree(RESULT_ROOT)

    def test_single_transcript_single_exon(self):
        self._test('single_transcript_single_exon')

    def test_single_transcript_multiple_exon(self):
        self._test('single_transcript_multiple_exon')

    def test_merged_gene(self):
        self._test('merged_gene')

    def test_single_transcript_single_intron(self):
        self._test('single_transcript_single_intron')

    def test_single_transcript_five_end_intron(self):
        self._test('single_transcript_five_end_intron')

    def test_single_transcript_three_end_intron(self):
        self._test('single_transcript_three_end_intron')

    def test_multiple_transcript_multiple_exon(self):
        self._test('multiple_transcript_multiple_exon',calculate_base=False)
