import unittest
import tensorflow as tf
import keras
from . import LossFactory
class TestLostFactory(unittest.TestCase):
    def test_perfect(self):
        loss_factory = LossFactory()
        loss = loss_factory.create()
        init = tf.global_variables_initializer() 
        predict = tf.constant([[[1,0,0],[1,0,0]],[[1,0,0],[1,0,0]]],dtype=tf.float32)
        real = tf.constant([[[1,0,0],[1,0,0]],[[1,0,0],[1,0,0]]],dtype=tf.float32)
        answer = tf.constant(1e-7,dtype=tf.float32)
        loss_value = loss(real,predict)
        with tf.Session() as sess:
            sess.run(init)
            self.assertAlmostEqual(sess.run(loss_value),sess.run(answer),5)
    def test_not_perfect(self):
        loss_factory = LossFactory()
        loss = loss_factory.create()
        init = tf.global_variables_initializer() 
        predict = tf.constant([[[.9,0,.1],[12,0,0]],[[1,0,0],[1,0,0]]],dtype=tf.float32)
        real = tf.constant([[[0,0,1],[0,0,1]],[[0,0,1],[0,0,1]]],dtype=tf.float32)
        answer = tf.constant(1e-7,dtype=tf.float32)
        loss_value = loss(real,predict)
        with tf.Session() as sess:
            sess.run(init)
            self.assertGreaterEqual(sess.run(loss_value),sess.run(answer))
    def test_balanced_value(self):
        loss_factory = LossFactory()
        loss = loss_factory.create()
        init = tf.global_variables_initializer() 
        predict = tf.constant([[[.9,0.1],[.5,.5]],[[.0,1],[.6,.4]]],dtype=tf.float32)
        real = tf.constant([[[1,0],[0,1]],[[0,1],[1,0]]],dtype=tf.float32)
        answer = tf.constant(0.32733336,dtype=tf.float32)
        loss_value = loss(real,predict)
        with tf.Session() as sess:
            sess.run(init)
            self.assertAlmostEqual(sess.run(loss_value),sess.run(answer),5)
    def test_balanced_value_with_weighted(self):
        loss_factory = LossFactory()
        loss = loss_factory.create(dynamic_weight_method="reversed_count_weight")
        init = tf.global_variables_initializer() 
        predict = tf.constant([[[.9,0.1],[.5,.5]],[[.0,1],[.6,.4]]],dtype=tf.float32)
        real = tf.constant([[[1,0],[0,1]],[[0,1],[1,0]]],dtype=tf.float32)
        answer = tf.constant(0.32733336,dtype=tf.float32)
        loss_value = loss(real,predict)
        with tf.Session() as sess:
            sess.run(init)
            self.assertAlmostEqual(sess.run(loss_value),sess.run(answer),5)
    def test_unbalanced_value(self):
        loss_factory = LossFactory()
        loss = loss_factory.create()
        init = tf.global_variables_initializer() 
        predict = tf.constant([[[.9,0.1],[.5,.5]],[[.0,1],[.6,.4]]],dtype=tf.float32)
        real = tf.constant([[[1,0],[1,0]],[[0,1],[1,0]]],dtype=tf.float32)
        answer = tf.constant(0.32733336,dtype=tf.float32)
        loss_value = loss(real,predict)
        with tf.Session() as sess:
            sess.run(init)
            self.assertAlmostEqual(sess.run(loss_value),sess.run(answer),5)
    def test_unbalanced_value_with_weighted(self):
        loss_factory = LossFactory()
        loss = loss_factory.create(dynamic_weight_method="reversed_count_weight")
        init = tf.global_variables_initializer() 
        predict = tf.constant([[[.9,0.1],[.5,.5]],[[.0,1],[.6,.4]]],dtype=tf.float32)
        real = tf.constant([[[1,0],[1,0]],[[0,1],[1,0]]],dtype=tf.float32)
        answer = tf.constant(0.32733336,dtype=tf.float32)
        loss_value = loss(real,predict)
        with tf.Session() as sess:
            sess.run(init)
            self.assertNotEqual(sess.run(loss_value),sess.run(answer))
if __name__=="__main__":
    unittest.TestSuite()
    unittest.TestLoader().loadTestsFromTestCase(TestLostFactory)
    unittest.main()
