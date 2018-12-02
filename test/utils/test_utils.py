import unittest
from sequence_annotation.utils.utils import process_tensor
import tensorflow as tf
import numpy as np
class TestUtils(unittest.TestCase):
    def test_process_tensor_with_ignore(self):
        init = tf.global_variables_initializer() 
        answer_ = tf.constant([[[1,0,0],[1,0,0]],[[0,0,0],[0,1,0]]],dtype=tf.float32)
        pred_ = tf.constant([[[1,0,0],[0,1,0]],[[1,0,0],[1,0,0]]],dtype=tf.float32)
        answer, pred = process_tensor(answer_, pred_,0)
        with tf.Session() as sess:
            sess.run(init)
            pred_val = sess.run(pred)
            answer_val = sess.run(answer)
            self.assertTrue(np.array_equal(answer_val,[[1,0,0],[1,0,0],[0,1,0]]))
            self.assertTrue(np.array_equal(pred_val,[[1,0,0],[0,1,0],[1,0,0]]))

    def test_process_tensor_without_ignore(self):
        init = tf.global_variables_initializer() 
        answer_ = tf.constant([[[1,0,0],[1,0,0]],[[0,0,0],[0,1,0]]],dtype=tf.float32)
        pred_ = tf.constant([[[1,0,0],[0,1,0]],[[1,0,0],[1,0,0]]],dtype=tf.float32)
        answer, pred = process_tensor(answer_, pred_)
        with tf.Session() as sess:
            sess.run(init)
            pred_val = sess.run(pred)
            answer_val = sess.run(answer)
            self.assertTrue(np.array_equal(answer_val,[[1,0,0],[1,0,0],[0,0,0],[0,1,0]]))
            self.assertTrue(np.array_equal(pred_val,[[1,0,0],[0,1,0],[1,0,0],[1,0,0]]))