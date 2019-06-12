#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:51:20 2019

@author: spikezz
"""

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 4], 'x')

#y = tf.keras.layers.BatchNormalization()(x, training=True)
y = tf.layers.batch_normalization(x, training=True)
p_y=tf.Print(y,[y],message='py:',summarize=1024)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

y_out = sess.run(p_y, feed_dict={x: [[20, 10, 0, 30]]})
#y_out = sess.run(p_y, feed_dict={x: [[0]]})
sess.close()

#is_traing = tf.placeholder(dtype=tf.bool)
#input = tf.ones([1, 2, 2, 3])
#output = tf.layers.batch_normalization(input, training=is_traing)
#
#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#print(update_ops)
## with tf.control_dependencies(update_ops):
#    # train_op = optimizer.minimize(loss)
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    saver = tf.train.Saver()
#    saver.save(sess, "batch_norm_layer/Model")

