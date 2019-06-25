#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:51:20 2019

@author: spikezz
"""

import tensorflow as tf

#x = tf.placeholder(tf.float32, [None, 4], 'x')

#y = tf.keras.layers.BatchNormalization()(x, training=True)
#y = tf.layers.batch_normalization(x, training=True)
#p_y=tf.Print(y,[y],message='py:',summarize=1024)
#
#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#
#y_out = sess.run(p_y, feed_dict={x: [[20, 10, 0, 30]]})
#y_out = sess.run(p_y, feed_dict={x: [[0]]})
#sess.close()

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

import tensorflow as tf

def _build_BN_layer(layer_input,layer_normalizer_scope,moving_decay=0.9,eps=1e-16,training_phase=0):
        
    with tf.variable_scope(layer_normalizer_scope):
        
        gamma = tf.get_variable('gamma',shape=[layer_input.shape[-1]],initializer=tf.constant_initializer(1))
        beta  = tf.get_variable('beat', shape=[layer_input.shape[-1]],initializer=tf.constant_initializer(0))
        
        axises = list(range(len(layer_input.shape)-1))
        batch_mean, batch_var = tf.nn.moments(layer_input,axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(moving_decay)   
        
        def mean_var_with_update():
            
            ema_apply_op = ema.apply([batch_mean,batch_var])
            
            with tf.control_dependencies([ema_apply_op]):
                
                return tf.identity(batch_mean), tf.identity(batch_var)
    
        mean, var = tf.cond(tf.equal(training_phase,0),mean_var_with_update,lambda:(ema.average(batch_mean),ema.average(batch_var)))
    
    return tf.nn.batch_normalization(layer_input,mean,var,beta,gamma,eps)

# 实现Batch Normalization
def bn_layer(x,is_training,name='BatchNorm',moving_decay=0.9,eps=1e-5):
    # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
    shape = x.shape
    assert len(shape) in [2,4]

    param_shape = shape[-1]
    with tf.variable_scope(name):
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        gamma = tf.get_variable('gamma',param_shape,initializer=tf.constant_initializer(1))
        beta  = tf.get_variable('beat', param_shape,initializer=tf.constant_initializer(0))

        # 计算当前整个batch的均值与方差
        axes = list(range(len(shape)-1))
        batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')

        # 采用滑动平均更新均值与方差
        ema = tf.train.ExponentialMovingAverage(moving_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
                lambda:(ema.average(batch_mean),ema.average(batch_var)))

        # 最后执行batch normalization
        return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)

# 注意bn_layer中滑动平均的操作导致该层只支持半精度、float32和float64类型变量
x = tf.constant([[1,2,1],[2,1,2],[0,1,0]],dtype=tf.float32)
y = _build_BN_layer(x,layer_normalizer_scope='BN')
y = bn_layer(x,True)

# 注意bn_layer中的一些操作必须被提前初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print('x = ',x.eval())
    print('y = ',y.eval())



