import torch
import unittest
from sequence_annotation.process.metric import categorical_metric, precision, recall, F1, calculate_metric, contagion_matrix


class TestMetric(unittest.TestCase):
    def test_categorical_metric(self):
        predict = torch.FloatTensor([[[1, 0, 0, 1, 0], [0, 0, 1, 0, 1],
                                      [0, 1, 0, 0, 0]]])
        answer = torch.FloatTensor([[[1, 0, 1, 0, 0], [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        result = categorical_metric(predict, answer, mask)
        self.assertEqual(1.0, result['T'])
        self.assertEqual(2.0, result['F'])
        self.assertEqual([1.0, 0.0, 0.0], result['TPs'])
        self.assertEqual([0.0, 1.0, 1.0], result['FPs'])
        self.assertEqual([1.0, 1.0, 2.0], result['TNs'])
        self.assertEqual([1.0, 1.0, 0.0], result['FNs'])

    def test_precision(self):
        result = precision([1.0, 0.0, 0.0], [0.0, 1.0, 1.0])
        self.assertEqual([1.0, 0.0, 0.0], result)

    def test_recall(self):
        result = recall([1.0, 0.0, 0.0], [1.0, 1.0, 0.0])
        self.assertEqual([0.5, 0.0, 0.0], result)

    def test_F1(self):
        result = F1([1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0])
        self.assertEqual([2 / 3, 0.0, 0.0], result)

    def test_calculate_metric(self):
        data = {
            'T': 1,
            'F': 2,
            'TPs': [1.0, 0.0, 0.0],
            'FPs': [0.0, 1.0, 1.0],
            'TNs': [1.0, 1.0, 2.0],
            'FNs': [1.0, 1.0, 0.0]
        }
        result = calculate_metric(data, round_value=2)
        self.assertEqual(1.0, result['precision_0'])
        self.assertEqual(0.0, result['precision_1'])
        self.assertEqual(0.0, result['precision_2'])
        self.assertEqual(0.5, result['recall_0'])
        self.assertEqual(0.0, result['recall_1'])
        self.assertEqual(0.0, result['recall_2'])
        self.assertEqual(.33, result['accuracy'])
        self.assertEqual(.67, result['F1_0'])
        self.assertEqual(0.0, result['F1_1'])
        self.assertEqual(0.0, result['F1_2'])
        self.assertEqual(.33, result['macro_precision'])
        self.assertEqual(.17, result['macro_recall'])
        self.assertEqual(0.22, result['macro_F1'])

    def test_contagion_matrix(self):
        predict = torch.FloatTensor([[[1, 0, 0, 1, 0], [0, 0, 1, 0, 1],
                                      [0, 1, 0, 0, 0]]])
        answer = torch.FloatTensor([[[1, 0, 1, 0, 0], [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        result = contagion_matrix(predict, answer, mask)
        self.assertEqual([[1, 1, 0], [0, 0, 1], [0, 0, 0]], result)
