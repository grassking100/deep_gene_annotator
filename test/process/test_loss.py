import torch
import unittest
from sequence_annotation.process.loss import sum_by_mask, mean_by_mask, bce_loss, CCELoss, FocalLoss, SeqAnnLoss


class TestLoss(unittest.TestCase):
    def test_sum_by_mask(self):
        data = torch.FloatTensor([[1, 1, 1, 1, 1]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        result = sum_by_mask(data, mask).item()
        self.assertEqual(3.0, result)

    def test_mean_by_mask(self):
        data = torch.FloatTensor([[1, 1, 1, 1, 1]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        result = mean_by_mask(data, mask).item()
        self.assertEqual(1.0, result)

    def test_bce_loss_with_mask(self):
        predict = torch.FloatTensor([[1, 0.5, 1, 1, 0.5]])
        answer = torch.FloatTensor([[1, 0, 1, 1, 1]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        result = bce_loss(predict, answer, mask=mask).item()
        self.assertEqual(0.23105, round(result, 5))

    def test_bce_loss(self):
        predict = torch.FloatTensor([[1, 0.5, 1, 1, 0.5]])
        answer = torch.FloatTensor([[1, 0, 1, 1, 1]])
        result = bce_loss(predict, answer).item()
        self.assertEqual(0.27726, round(result, 5))

    def test_focal_loss(self):
        predict = torch.FloatTensor([[[1, 0.5, 1, 1, 0.5], [0, 0.5, 0, 0, 0.5],
                                      [0, 0, 0, 0, 0]]])
        answer = torch.FloatTensor([[[1, 0, 1, 0, 0], [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        loss = FocalLoss(gamma=2)
        result = loss(predict, answer, mask).item()
        self.assertEqual(0.05776, round(result, 5))

    def test_categorical_cross_entropy(self):
        predict = torch.FloatTensor([[[1, 0.5, 1, 1, 0.5], [0, 0.5, 0, 0, 0.5],
                                      [0, 0, 0, 0, 0]]])
        answer = torch.FloatTensor([[[1, 0, 1, 0, 0], [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        loss = CCELoss()
        result = loss(predict, answer, mask).item()
        self.assertEqual(0.23105, round(result, 5))

    def test_categorical_cross_entropy_padded_one(self):
        predict = torch.FloatTensor([[[1, 0.5, 1, 1, 0.5], [0, 0.5, 0, 0, 0.5],
                                      [0, 0, 0, 0, 0]]])
        answer = torch.FloatTensor([[[1, 0, 1, 1, 1], [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        loss = CCELoss()
        result = loss(predict, answer, mask).item()
        self.assertEqual(0.23105, round(result, 5))

    def test_accumulated_categorical_cross_entropy(self):
        predict = torch.FloatTensor([[[1, 0.5, 1, 1, 0.5], [0, 0.5, 0, 0, 0.5],
                                      [0, 0, 0, 0, 0]]])
        answer = torch.FloatTensor([[[1, 0, 1, 0, 0], [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]])
        mask = torch.FloatTensor([[1, 1, 1, 0, 0]])
        predict_2 = torch.FloatTensor([[[1, 0.5], [0, 0.5], [0, 0]]])
        answer_2 = torch.FloatTensor([[[1, 0], [0, 1], [0, 0]]])
        mask_2 = torch.FloatTensor([[1, 1]])
        loss = CCELoss()
        loss(predict, answer, mask, accumulate=True)
        result = loss(predict_2, answer_2, mask_2, accumulate=True).item()
        self.assertEqual(0.27726, round(result, 5))

    def test_seq_ann_loss(self):
        predict = torch.FloatTensor([[[0, 0.5, 1, 0, 0.0, 0, 0],
                                      [0, 0.5, 0, 1, 0.5, 0, 0]]])

        answer = torch.FloatTensor([[[0, 1, 0, 1, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1, 0, 0]]])

        mask = torch.FloatTensor([[1, 1, 1, 1, 1, 0, 0]])
        loss = SeqAnnLoss()
        result = loss(predict, answer, mask).item()
        self.assertEqual(12.15828, round(result, 5))

    def test_accumulated_seq_ann_loss(self):
        predict = torch.FloatTensor([[[0, 0.5, 1, 0], [0, 0.5, 0, 1]]])

        predict_2 = torch.FloatTensor([[[0, 0, 0], [.5, 0, 0]]])

        answer = torch.FloatTensor([[[0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0,
                                                                  0]]])

        answer_2 = torch.FloatTensor([[[0, 0, 0], [0, 0, 0], [1, 0, 0]]])

        mask = torch.FloatTensor([[1, 1, 1, 1]])
        mask_2 = torch.FloatTensor([[1, 0, 0]])
        loss = SeqAnnLoss()
        loss(predict, answer, mask, accumulate=True)
        result = loss(predict_2, answer_2, mask_2, accumulate=True).item()
        self.assertEqual(12.15828, round(result, 5))
