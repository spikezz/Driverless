#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:37:21 2019

@author: spikezz
"""

#import itertools

import tensorflow as tf
import numpy as np

tf_seed=1
np.random.seed(3)
tf.set_random_seed(1)


class Actor_Imitation(object):
    
    def __init__(self,agent,input_dim,action_dim,sess,action_boundary,lr_i,training_phase=False):
        
        with tf.name_scope('State_Imitation_Old'):
            
            self.S = tf.placeholder(tf.float32, shape=[None, input_dim], name='s')
            
        with tf.name_scope('Action_Controller'):
            
            self.A_C = tf.placeholder(tf.float32, [None, action_dim], name='a_c')
        
#        self.H_a=[2780,2780]    
#        self.H_a=[2780,1450,110]
#        self.H_a=[1390,750,110]
#        self.H_a=[1390,1390,1390]
        self.H_a=[2780,1890,1000,110]
#        self.H_a=[1390,970,540,110]
#        self.H_a=[970,690,400,110]
#        self.H_a=[1390,1070,750,430,110]
        self.momentum = 0.9
        self.std_decay=tf.constant(0.99999)
        self.sess=sess
        self.state_dimension = input_dim
        self.action_dimension = action_dim
        self.action_boundary = action_boundary
        self.lr_i = lr_i
        self.number_of_hidden_layers=len(self.H_a)
        
        with tf.variable_scope('actor_imitation'):
            
            self.a = self._build_net(agent,self.S,dropout_rate=0.05, batch_normalization=False,summeraize_parameter=False,\
                                     summeraize_output=True, trainable=True,training_phase=training_phase)
            
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_imitation')
            
        self.loss_imitation=tf.reduce_mean(tf.squared_difference(self.a, self.A_C))
        self.loss_imitation_scalar=tf.summary.scalar('loss_imitation', self.loss_imitation)
        agent.summary_set.append(self.loss_imitation_scalar)
            
        with tf.variable_scope('imitation_train'):
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            with tf.control_dependencies(update_ops):

                self.opt_i = tf.train.AdamOptimizer(self.lr_i)
#                self.opt_i = tf.train.MomentumOptimizer(self.lr_i,self.momentum)  
                self.i_grads = tf.gradients(ys=self.loss_imitation, xs=self.e_params, grad_ys=None)
                self.i_grads_filter=[]
                self.e_params_filter=[]
#              
                
                for i in range(0,len(self.i_grads)):
                    
                    if self.i_grads[i]!=None:
                        
                        self.i_grads_filter.append(self.i_grads[i])
                        self.e_params_filter.append(self.e_params[i])
                        self.i_grads_hist=tf.summary.histogram('i_grads_%d'%(i),self.i_grads[i])
                        agent.summary_set.append(self.i_grads_hist)

                self.train_imitation=self.opt_i.apply_gradients(zip(self.i_grads,self.e_params))
#                self.train_imitation=self.opt_i.apply_gradients(zip(self.i_grads_filter,self.e_params_filter))

      

    def _build_BN_layer(self,layer_input,layer_normalizer_scope,moving_decay=0.9,eps=1e-16,training_phase=0,trainable=True):
        
        with tf.variable_scope(layer_normalizer_scope):
            
            gamma = tf.get_variable('gamma',shape=[layer_input.shape[-1]],initializer=tf.constant_initializer(1),trainable=trainable)
            beta  = tf.get_variable('beat', shape=[layer_input.shape[-1]],initializer=tf.constant_initializer(0),trainable=trainable)
            
            axises = list(range(len(layer_input.shape)-1))
            batch_mean, batch_var = tf.nn.moments(layer_input,axises, name='moments')
            ema = tf.train.ExponentialMovingAverage(moving_decay)   
            
            def mean_var_with_update():
                
                ema_apply_op = ema.apply([batch_mean,batch_var])
                
                with tf.control_dependencies([ema_apply_op]):
                    
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(tf.equal(training_phase,0),mean_var_with_update,lambda:(ema.average(batch_mean),ema.average(batch_var)))

        return tf.nn.batch_normalization(layer_input,mean,var,beta,gamma,eps)
    
    def _build_layer(self,agent, layer_input_dim, hidden_neuro_dim, layer_input,layer_idx,initializer_w,initializer_b, dropout_rate=0,\
                    batch_normalization=False,summeraize_parameter=False, summeraize_output=False, trainable=None,training_phase=0): 
        
        layer_scope = 'layer_%d'%(layer_idx+1)
        weight_scope = 'weight_actor_layer%d'%(layer_idx+1)
        bias_scope = 'bias_actor_layer%d'%(layer_idx+1)
        layer_weight_histogram_scope='hist_weight_actor_layer%d'%(layer_idx+1)
        layer_bias_histogram_scope='hist_bias_actor_layer%d'%(layer_idx+1)   
        layer_normalizer_scope='normalizer_actor_layer%d'%(layer_idx+1)  
        layer_output_scope='output_actor_layer%d'%(layer_idx+1)
        
        with tf.variable_scope(layer_scope):
            
            w_collection = tf.get_variable(weight_scope, [layer_input_dim, hidden_neuro_dim], initializer=initializer_w, trainable=trainable)
            b_collection = tf.get_variable(bias_scope, [1, hidden_neuro_dim], initializer=initializer_b, trainable=trainable)
            
            if summeraize_parameter==True:
                    
                l_w_hist=tf.summary.histogram(layer_weight_histogram_scope,w_collection)
                agent.summary_set.append(l_w_hist)
                l_b_hist=tf.summary.histogram(layer_bias_histogram_scope,b_collection)
                agent.summary_set.append(l_b_hist) 
           
            layer_output_0 = tf.matmul(layer_input, w_collection) + b_collection  
            
            if batch_normalization:
                
                layer_output_normalization=self._build_BN_layer(layer_output_0,layer_normalizer_scope,trainable=True)
                layer_output = tf.nn.leaky_relu(layer_output_normalization)
                
            else:
                
                layer_output=tf.nn.leaky_relu(layer_output_0)
                
                
            layer_output_dropout = tf.nn.dropout(layer_output,rate=dropout_rate) 
            
            if summeraize_output==True:
                  
                lo_hist=tf.summary.histogram(layer_output_scope,layer_output_dropout)
                agent.summary_set.append(lo_hist) 
                
            return layer_output_dropout
        
    def _build_net(self, agent,s, dropout_rate=0,batch_normalization=False, summeraize_parameter=False,\
                   summeraize_output=False, trainable=False,training_phase=0):
    
        init_w = tf.keras.initializers.glorot_normal(seed=tf_seed)
        init_b = tf.keras.initializers.glorot_normal(seed=tf_seed)
        
        layer=[None]*(self.number_of_hidden_layers)
        
        layer[0]=self._build_layer(agent,self.state_dimension,self.H_a[0], s, 0, init_w, init_b, dropout_rate=dropout_rate,\
             batch_normalization=batch_normalization,summeraize_parameter=summeraize_parameter,summeraize_output=summeraize_output,trainable=trainable,training_phase=training_phase)

        for i in range(1,self.number_of_hidden_layers):
            
            layer[i]=self._build_layer(agent,self.H_a[i-1],self.H_a[i], layer[i-1], i, init_w, init_b, dropout_rate=dropout_rate,\
                 batch_normalization=batch_normalization,summeraize_parameter=summeraize_parameter,summeraize_output=summeraize_output,trainable=trainable,training_phase=training_phase)
        
        with tf.variable_scope('action'):
            
            w_collection_action = tf.get_variable('weight_actor_action', [self.H_a[-1], self.action_dimension], initializer=init_w, trainable=trainable)
            #leCun Tanh
#            action= 1.7519 * tf.nn.tanh(2*tf.matmul(layer[-1], w_collection_action)/3)
            action= tf.nn.tanh(tf.matmul(layer[-1], w_collection_action))
#            action= tf.nn.tanh(tf.matmul(layer[-1], w_collection_action)/5)
#            action= tf.matmul(layer[-1], w_collection_action)
            if summeraize_parameter==True:
                
                a_w_hist=tf.summary.histogram('hist_weight_action',w_collection_action)
                agent.summary_set.append(a_w_hist)
            
            if summeraize_output==True:
         
                actions_hist=tf.summary.histogram('hist_action',action)
                agent.summary_set.append(actions_hist)
        
            scaled_action = tf.multiply(action, self.action_boundary, name='scaled_action')  # Scale output to -action_bound to action_bound
   
        return scaled_action  
        
    def learn_Imitation(self, s ,a_c):   
        
        self.sess.run(self.train_imitation, feed_dict={self.S: s,self.A_C:a_c})

    def choose_action(self, s):
        
        s = s[np.newaxis, :]   
        action=self.sess.run(self.a,feed_dict={self.S: s})

        return action[0]#reduce dimension
            