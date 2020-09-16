import torch
import unittest
import numpy as np
from sequence_annotation.utils.metric import get_categorical_metric,  get_confusion_matrix, MetricCalculator
from sequence_annotation.utils.metric import calculate_precision, calculate_recall, calculate_F1

class TestMetric(unittest.TestCase):
    def test_categorical_metric(self):
        predict = torch.FloatTensor([[[1, 0, 0, 1, 0], [0, 0, 1, 0, 1],
                                      [0, 1, 0, 0, 0]]])
        answer = torch.FloatTensor([[[1, 0, 1, 0, 0], [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        result = get_categorical_metric(predict, answer, mask)
        self.assertEqual([1.0, 0.0, 0.0], result['TP'])
        self.assertEqual([0.0, 1.0, 1.0], result['FP'])
        self.assertEqual([1.0, 1.0, 0.0], result['FN'])

    def test_precision(self):
        result = calculate_precision([1.0, 0.0, 0.0], [0.0, 1.0, 1.0])
        self.assertEqual([1.0, 0.0, 0.0], result)

    def test_recall(self):
        result = calculate_recall([1.0, 0.0, 0.0], [1.0, 1.0, 0.0],True)
        self.assertEqual([0.5, 0.0, 0.0], result)

    def test_F1(self):
        result = calculate_F1([1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0])
        self.assertEqual([2 / 3, 0.0, 0.0], result)

    def test_calculate_metric(self):
        data = {
            'TP': [1.0, 0.0, 0.0],
            'FP': [0.0, 1.0, 1.0],
            'FN': [1.0, 1.0, 0.0]
        }
        calculator = MetricCalculator(3,round_value=2,precision=True,recall=True)
        result = calculator(data)
        self.assertEqual(1.0, result['precision_0'])
        self.assertEqual(0.0, result['precision_1'])
        self.assertEqual(0.0, result['precision_2'])
        self.assertEqual(0.5, result['recall_0'])
        self.assertEqual(0.0, result['recall_1'])
        self.assertEqual(0.0, result['recall_2'])
        self.assertEqual(.67, result['F1_0'])
        self.assertEqual(0.0, result['F1_1'])
        self.assertEqual(0.0, result['F1_2'])
        self.assertEqual(.33, result['macro_precision'])
        self.assertEqual(.17, result['macro_recall'])
        self.assertEqual(0.22, result['macro_F1'])

    def test_confusion_matrix(self):
        predict = torch.FloatTensor([[[1, 0, 0, 1, 0], [0, 0, 1, 0, 1],
                                      [0, 1, 0, 0, 0]]])
        answer = torch.FloatTensor([[[1, 0, 1, 0, 0], [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        result = get_confusion_matrix(predict, answer, mask)
        np.testing.assert_array_equal([[1, 1, 0], [0, 0, 1], [0, 0, 0]], result)
