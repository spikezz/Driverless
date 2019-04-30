#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:06:58 2019

@author: spikezz
"""
import itertools
import tensorflow as tf
import numpy as np

#import os
#from keras import regularizers

np.random.seed(3)
tf.set_random_seed(1)
tf_seed=1

H1_a=4400  
H2_a=4400 
H3_a=4400 
H4_a=4400
H1_c=140
H2_c=140
H3_c=140
input_dim = 100
action_dim = 2
TAU_A=0.001
TAU_C=0.001

with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, input_dim], name='s')
with tf.name_scope('R0'):
    R0 = tf.placeholder(tf.float32, [None, 1], name='r0')
with tf.name_scope('R1'):
    R1 = tf.placeholder(tf.float32, [None, 1], name='r1')
with tf.name_scope('R2'):
    R2 = tf.placeholder(tf.float32, [None, 1], name='r2')
with tf.name_scope('A_I'):
    A_I = tf.placeholder(tf.float32, [None, action_dim], name='a_i')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, input_dim], name='s_')

class Actor(object):
    
    def __init__(self, sess, action_dimension, state_dimension, action_bound, lr_r=None, lr_i=None, t_replace_iter=None, agent_i=False, agent_r=False):
        
        self.sess = sess
        
        self.s_dim = state_dimension
        self.a_dim = action_dimension
        
        self.action_bound = action_bound
        
        self.lr_r = lr_r
        self.lr_i = lr_i
        
        self.momentum = 0.9
        
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        
        self.number_of_hidden_layers=4
        
        self.steering_angle_controller=[]
        self.steering_angle_imitation=[]
        self.throttle_controller=[]
        self.throttle_imitation=[]
        self.brake_controller=[]
        self.brake_imitation=[]
        
        self.writer = tf.summary.FileWriter("/home/spikezz/RL project copy/Driverless/src_c/logs")
        
        self.summary_set=[]
        
        self.std_decay=tf.constant(0.99999)
        
        if agent_i:
            
            self.scope_name='actor_imitation'
            
        if agent_r:
            
            self.scope_name='actor_reinforcement'
            
        with tf.variable_scope(self.scope_name):
            
            # input s, output a    
            self.a = self._build_net(S, parameter_noise=False, summeraize_parameter=False,summeraize_output=False,\
                                     scope='evaluation_net', trainable=True)
            
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name+'/evaluation_net')

            if agent_i:
                
                self.loss_imitation=tf.reduce_mean(tf.squared_difference(self.a, A_I))
#            #cross entropy
#            self.a_distribution=tf.nn.softmax(self.a) 
#            self.a_i_distribution=tf.nn.softmax(A_I) 
#            self.loss_imitation=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits =self.a_distribution, labels = self.a_i_distribution)) 
#            self.print_loss_imitation=tf.Print(self.loss_imitation,[self.a,A_I,self.a_distribution,self.a_i_distribution,self.loss_imitation],message='loss_imitation', summarize=32)
                
                self.loss_imitation_scalar=tf.summary.scalar('loss_imitation', self.loss_imitation)
                self.summary_set.append(self.loss_imitation_scalar)
                
                with tf.variable_scope('imitation_train'):
                    
        #            self.train_op = tf.train.AdamOptimizer(self.lr_i).minimize(self.loss_imitation)
        #            self.train_imitation = tf.train.MomentumOptimizer(self.lr_i,self.Momentum).minimize(self.loss_imitation)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    update_ops = list(itertools.chain.from_iterable(update_ops))
#                    print(update_ops)
                    with tf.control_dependencies(update_ops):
                        
                        self.opt_i = tf.train.MomentumOptimizer(self.lr_i,self.momentum)     
                        self.i_grads = tf.gradients(ys=self.loss_imitation, xs=self.e_params, grad_ys=None)
                        self.i_grads_hist=tf.summary.histogram('i_grads',self.i_grads[0])
                        self.summary_set.append(self.i_grads_hist)     
                        self.train_imitation=self.opt_i.apply_gradients(zip(self.i_grads,self.e_params))
            
            if agent_r:
                            # input s_, output a_, get a_ for critic
                self.a_ = self._build_net(S_, parameter_noise=False, summeraize_parameter=False,summeraize_output=False,\
                                      scope='target_net', trainable=False)
                
                self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name+'/target_net')
                
                with tf.variable_scope('noise_standard_deviation_minmax'):
                    
                    self.init_noise_std=[0]*self.number_of_hidden_layers
                    self.layer_noise_std=[0]*self.number_of_hidden_layers
                    self.noise_min=[0]*self.number_of_hidden_layers
                    self.noise_max=[0]*self.number_of_hidden_layers
                    
                    for i in range(0,self.number_of_hidden_layers):
                        
                        self.init_noise_std[i]=tf.constant_initializer(0.002) 
                        self.layer_noise_std[i]=tf.get_variable('w%d_s_noise_std'%(i+1),[],initializer=self.init_noise_std[i], trainable=False)
                        self.noise_min[i]=tf.constant_initializer(-0.002) 
                        self.noise_max[i]=tf.constant_initializer(0.002) 
                
                self.a_noise = self._build_net(S_, parameter_noise=True, summeraize_parameter=False,summeraize_output=False,\
                                      scope='evaluation_net_with_parameter_noise', trainable=False)
                self.error_parameter_noise=tf.reduce_mean(tf.squared_difference(self.a, self.a_noise))
                
                self.error_scalar=tf.summary.scalar('parameter_noise_action_error', self.error_parameter_noise)
                self.summary_set.append(self.error_scalar)
                    
                self.e_params_with_noise = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name+'/evaluation_net_with_parameter_noise')
                self.parameter_copy_noise_net = [tf.assign(e_n, e)for e_n, e in zip(self.e_params_with_noise, self.e_params)]
                self.soft_replace = [tf.assign(t, (1 - TAU_A) * t + TAU_A * e) for t, e in zip(self.t_params, self.e_params)]

    def duplicate_net(self):

        self.sess.run(self.parameter_copy_noise_net)        
        
    def _build_layer(self, hidden_neuro_dim, layer_input_dim,layer_input,layer_idx,initializer_w,initializer_b, dropout_rate=0.1, \
                     parameter_noise=False,summeraize_parameter=False, summeraize_output=False, trainable=None): 
        if parameter_noise:
            
            layer_scope = 'layer_%d_with_parameter_noise'%(layer_idx+1)
           
        else:
            
            layer_scope = 'layer_%d'%(layer_idx+1)
        
        weight_scope = 'weight_actor_layer%d'%(layer_idx+1)
        bias_scope = 'bias_actor_layer%d'%(layer_idx+1)
        
        layer_weight_histogram_scope='hist_weight_actor_layer%d'%(layer_idx+1)
        layer_bias_histogram_scope='hist_bias_actor_layer%d'%(layer_idx+1)
        
        layer_normalization_scope='normalization_actor_layer%d'%(layer_idx+1)
        layer_normalizer_scope='normalizer_actor_layer%d'%(layer_idx+1)
        
        layer_noise_scope='actor_layer%d_with_noise'%(layer_idx+1)
        
        weight_noise_scope='weight%d_with_noise'%(layer_idx+1)
        bias_noise_scope='bias%d_with_noise'%(layer_idx+1)
        
        layer_weight_noise_histogram_scope='hist_weight_with_noise_actor_layer%d'%(layer_idx+1)
        layer_bias_noise_histogram_scope='hist_bias_with_noise_actor_layer%d'%(layer_idx+1)
        
        layer_output_scope='output_actor_layer%d'%(layer_idx+1)
        
        with tf.variable_scope(layer_scope):
            
            with tf.variable_scope(layer_normalization_scope):
                
                batch_normalization=tf.keras.layers.BatchNormalization(name=layer_normalizer_scope)
                layer_input_normalization=batch_normalization(layer_input,training=True)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, batch_normalization.updates)
            
            w_collection = tf.get_variable(weight_scope, [layer_input_dim, hidden_neuro_dim], initializer=initializer_w, trainable=trainable)
            b_collection = tf.get_variable(bias_scope, [1, hidden_neuro_dim], initializer=initializer_b, trainable=trainable)
            
            if summeraize_parameter==True:
                    
                l_w_hist=tf.summary.histogram(layer_weight_histogram_scope,w_collection)
                self.summary_set.append(l_w_hist)
                l_b_hist=tf.summary.histogram(layer_bias_histogram_scope,b_collection)
                self.summary_set.append(l_b_hist) 
            
            if parameter_noise:
            
                with tf.variable_scope(layer_noise_scope):
                    
                    w_noise=tf.random_normal(w_collection.shape,stddev=self.layer_noise_std[layer_idx],name=weight_noise_scope)
                    w_collection_with_noise=w_collection+w_noise
                    
                    b_noise=tf.random_normal(b_collection.shape,stddev=self.layer_noise_std[layer_idx],name=bias_noise_scope)
                    b_collection_with_noise=b_collection+b_noise
                    
                    if summeraize_parameter==True:
                        
                        l_w_noise_hist=tf.summary.histogram(layer_weight_noise_histogram_scope,w_collection_with_noise)
                        self.summary_set.append(l_w_noise_hist)
                        l_b_noise_hist=tf.summary.histogram(layer_bias_noise_histogram_scope,b_collection_with_noise)
                        self.summary_set.append(l_b_noise_hist)
                    
                    layer_output_with_noise=tf.nn.leaky_relu(tf.matmul(layer_input_normalization, w_collection_with_noise)+b_collection_with_noise)   
                    layer_output_disturbed=tf.nn.dropout(layer_output_with_noise,rate=dropout_rate)
                                
                    if summeraize_output==True:
                    
                        l_disturbed_hist=tf.summary.histogram(layer_output_scope,layer_output_disturbed)
                        self.summary_set.append(l_disturbed_hist)
                        
                    return  layer_output_disturbed
                
            else:
                
                layer_output_without_noise = tf.nn.leaky_relu(tf.matmul(layer_input_normalization, w_collection) + b_collection)
                layer_output = tf.nn.dropout(layer_output_without_noise,rate=dropout_rate) 
                
                if summeraize_output==True:
                        
                    l_hist=tf.summary.histogram(layer_output_scope,layer_output)
                    self.summary_set.append(l_hist) 
                        
                return layer_output
                            
    def _build_net(self, s, parameter_noise=False, summeraize_parameter=False, summeraize_output=False, scope=None, trainable=None):
        
        with tf.variable_scope(scope):
            
#            init_w=tf.random_uniform_initializer(-0.9,0.9)
#            init_b=tf.random_uniform_initializer(-0.15,0.15)
#            init_w=tf.random_normal_initializer(0,0.05)
#            init_b=tf.random_normal_initializer(0,0.05)
#            init_w = tf.keras.initializers.he_normal()
#            init_b= tf.keras.initializers.he_normal()
            
            init_w = tf.keras.initializers.glorot_normal(seed=tf_seed)
            init_b = tf.keras.initializers.glorot_normal(seed=tf_seed)
                
            layer=[None]*(self.number_of_hidden_layers)
            
            layer[0]=self._build_layer(H1_a, self.s_dim, s, 0, init_w, init_b, dropout_rate=0.1, parameter_noise=parameter_noise, \
                     summeraize_parameter=summeraize_parameter,summeraize_output=summeraize_output,trainable=trainable)
            
            layer[1]=self._build_layer(H2_a, H1_a, layer[0], 1, init_w,init_b, dropout_rate=0.1, parameter_noise=parameter_noise, \
                     summeraize_parameter=summeraize_parameter,summeraize_output=summeraize_output,trainable=trainable)
            
            layer[2]=self._build_layer(H3_a, H2_a, layer[1], 2, init_w,init_b, dropout_rate=0.1, parameter_noise=parameter_noise, \
                     summeraize_parameter=summeraize_parameter,summeraize_output=summeraize_output,trainable=trainable)
            
            layer[3]=self._build_layer(H4_a, H3_a, layer[2], 3, init_w,init_b, dropout_rate=0.1, parameter_noise=parameter_noise, \
                     summeraize_parameter=summeraize_parameter,summeraize_output=summeraize_output,trainable=trainable)
            
            with tf.variable_scope('action'):
            
                w_collection_action = tf.get_variable('weight_actor_action', [H4_a, self.a_dim], initializer=init_w, trainable=trainable)
                actions= tf.nn.tanh(0.1*tf.matmul(layer[3], w_collection_action))  
                
                if summeraize_parameter==True:
                    
                    a_w_hist=tf.summary.histogram('hist_weight_action',w_collection_action)
                    self.summary_set.append(a_w_hist)
                
                if summeraize_output==True:
             
                    actions_hist=tf.summary.histogram('hist_action',actions)
                    self.summary_set.append(actions_hist)
            
                scaled_action = tf.multiply(actions, self.action_bound, name='scaled_action')  # Scale output to -action_bound to action_bound
       
            return scaled_action  
    
    def Merge_Summary_End(self,s,a_i,time_step,critic):
        
        try:
            self.merge_summary = tf.summary.merge(self.summary_set)
            self.summary_actor=self.sess.run(self.merge_summary,feed_dict={S: s, S_: s, A_I:a_i})
            critic.writer.add_summary(self.summary_actor,time_step)
        except:
            print("nothing is here")
        
    def learn(self, s):   # batch update
        self.sess.run(self.soft_replace)
        self.sess.run(self.train_op, feed_dict={S: s})
#        if self.t_replace_counter == self.t_replace_iter:
#            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
#            self.t_replace_counter = 0
#        self.t_replace_counter += 1
        
    def learn_Imitation(self, s,a_i):   # batch update
#        self.sess.run(self.soft_replace)
#        self.sess.run([self.train_imitation,self.print_loss_imitation], feed_dict={S: s,A_I:a_i})
        self.sess.run(self.train_imitation, feed_dict={S: s,A_I:a_i})
#        if self.t_replace_counter == self.t_replace_iter:
#            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
#            self.t_replace_counter = 0
#        self.t_replace_counter += 1
    def choose_action(self, s):
        
        s = s[np.newaxis, :]    # single state
#        _,a=self.sess.run([self.p_layer1_n_n,self.a],feed_dict={S: s,S_: s})
        a=self.sess.run(self.a,feed_dict={S: s})
        for i in range(0,self.number_of_hidden_layers):
            
            self.layer_noise_std[i]=tf.multiply(self.layer_noise_std[i], self.std_decay)

        return a[0]  # single action

    def add_grad_to_graph(self, a_grads):
        
        with tf.variable_scope('policy_grads'):
            
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=-a_grads)
#            self.actions_policy_grads_hist=tf.summary.histogram('actions_policy_grads_',self.policy_grads[0])
#            self.summary_set.append(self.actions_policy_grads_hist)
            
        with tf.variable_scope('A_train'):
            
            opt = tf.train.MomentumOptimizer(self.lr_r,self.momentum)# (- learning rate) for ascent policy
#            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    
    def __init__(self, sess, action_dim,state_dim, learning_rate, gamma ,a, a_,t_replace_iter,C_TD):
        
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.loss_step=0
#        self.one_ep_step=0
        self.Momentum=0.9
        self.epsilon=1e-16
        
        self.rank_reward_max=1
        self.rank_TD_max=1
        self.rank_reward_min=1
        self.rank_TD_min=1
        self.c_TD=C_TD
        self.TD_set = np.zeros(self.c_TD)
        self.TD_set_no_zero=np.array([])
        self.TD_step=0
        
        self.model_localization=[]
        self.summary_set=[]
#        self.summary_step=0

        self.writer = tf.summary.FileWriter("/home/spikezz/RL project copy/Driverless/src_c/logs")
        
        
        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q= self._build_net(S, self.a, 'eval_net', trainable=True)[0]

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False) [0]  # target_q is based on a_ from Actor's target_net

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')
        self.soft_replace = [tf.assign(t, (1 - TAU_C) * t + TAU_C * e) for t, e in zip(self.t_params, self.e_params)]
        
        with tf.variable_scope('target_q'):
            
            R_mean=[0,0,0]
            R_var=[0,0,0]
            r=[0,0,0]
#            P_R=tf.Print(R,[R],message="no normalized R",summarize=32)
            R_mean[0],R_var[0]=tf.nn.moments(R0,0)
            R_mean[1],R_var[1]=tf.nn.moments(R1,0)
            R_mean[2],R_var[2]=tf.nn.moments(R2,0)
#            P_m_v=tf.Print(P_R,[R_mean,R_var],message="R_mean,R_var")
            r[0]=tf.nn.batch_normalization(R0,R_mean[0],R_var[0],0,1,self.epsilon,name='speed_reward_normalization')
            r[1]=tf.nn.batch_normalization(R1,R_mean[1],R_var[1],0,1,self.epsilon,name='angle_reward_normalization')
            r[2]=tf.nn.batch_normalization(R2,R_mean[2],R_var[2],0,1,self.epsilon,name='distance_difference_reward_normalization')
#            P_r=tf.Print(P_m_v,[r],message="normalized r",summarize=32)
            r_syn=r[0]+r[1]+r[2]
            self.target_q = r_syn + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))
            self.loss_scalar=tf.summary.scalar('loss_', self.loss)
            self.summary_set.append(self.loss_scalar)
            
        with tf.variable_scope('C_train'):
#            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
#            self.train_op = tf.train.MomentumOptimizer(self.lr,self.Momentum).minimize(self.loss)
            
            self.opt = tf.train.MomentumOptimizer(self.lr,self.Momentum)      
            self.q_grads = tf.gradients(ys=self.loss, xs=self.e_params, grad_ys=None)
#            self.q_grads_hist=tf.summary.histogram('q_grads_',self.q_grads[0])
#            self.summary_set.append(self.q_grads_hist)
#            
            self.train_op=self.opt.apply_gradients(zip(self.q_grads,self.e_params))
            
        with tf.variable_scope('a_grad'):
            
            self.a_grads = 20*tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)
#            self.a_grads_hist=tf.summary.histogram('a_grads_',self.a_grads)
#            self.summary_set.append(self.a_grads_hist)
    
        
    def _build_net(self, s, a, scope, trainable):
        
        with tf.variable_scope(scope):
            
#            init_w=tf.random_uniform_initializer(-0.9,0.9)
#            init_b=tf.random_uniform_initializer(-0.15,0.15)
            init_w = tf.keras.initializers.glorot_normal(seed=None)
            init_b = tf.keras.initializers.glorot_normal(seed=None)
#            init_w=tf.random_normal_initializer(0,0.05)
#            init_b=tf.random_normal_initializer(0,0.05)
#            init_w = tf.keras.initializers.he_normal()
#            init_b= tf.keras.initializers.he_normal()
            
            with tf.variable_scope('l1'):
                
                n_l1 = H1_c
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                layer1 = tf.nn.leaky_relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
                
#                s_n=tf.layers.batch_normalization(s,name='s_normalize_c')
#                a_n=tf.layers.batch_normalization(a,name='a_normalize_c')
#                layer1 = tf.nn.leaky_relu(tf.matmul(s_n, w1_s) + tf.matmul(a_n, w1_a) + b1)
                
#                self.l1_ws_hist=tf.summary.histogram('l1_c_ws_',w1_s)
#                self.summary_set.append(self.l1_ws_hist)
#                self.l1_wa_hist=tf.summary.histogram('l1_c_wa_',w1_a)
#                self.summary_set.append(self.l1_wa_hist)
#                self.l1_b_hist=tf.summary.histogram('l1_c_b_',b1)
#                self.summary_set.append(self.l1_b_hist)
#                
#            with tf.variable_scope('l1_normalization'):
#                
#                layer1_n=tf.layers.batch_normalization(layer1,name='layer1_n_')
#                self.l1_hist=tf.summary.histogram('l1_c_n_',layer1_n)
#                self.summary_set.append(self.l1_hist)
                
            with tf.variable_scope('l2'):
                n_l2 = H2_c
                w2_s = tf.get_variable('w2_s', [n_l1, n_l2], initializer=init_w, trainable=trainable)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
                layer2 = tf.nn.leaky_relu(tf.matmul(layer1, w2_s) + b2)
#                layer2 = tf.nn.leaky_relu(tf.matmul(layer1_n, w2_s) + b2)
                
#                self.l2_w_hist=tf.summary.histogram('l2_c_w_',w2_s)
#                self.summary_set.append(self.l2_w_hist)
#                self.l2_b_hist=tf.summary.histogram('l2_c_b_',b2)
#                self.summary_set.append(self.l2_b_hist)
                

#            with tf.variable_scope('l2_normalization'):
#                
#                layer2_n=tf.layers.batch_normalization(layer2,name='layer2_n_')
#                self.l2_hist=tf.summary.histogram('l2_c_n_',layer2_n)
#                self.summary_set.append(self.l2_hist)
                
            with tf.variable_scope('l3'):
                n_l3 = H3_c
                w3_s = tf.get_variable('w3_s', [n_l2, n_l3], initializer=init_w, trainable=trainable)
                b3 = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
                layer3 = tf.nn.leaky_relu(tf.matmul(layer2, w3_s) + b3)
#                layer3 = tf.nn.leaky_relu(tf.matmul(layer2_n, w3_s) + b3)

#                self.l3_w_hist=tf.summary.histogram('l3_c_w_',w3_s)
#                self.summary_set.append(self.l3_w_hist)
#                self.l3_b_hist=tf.summary.histogram('l3_c_b_',b3)
#                self.summary_set.append(self.l3_b_hist)              

#            with tf.variable_scope('l3_normalization'):
#                
#                layer3_n=tf.layers.batch_normalization(layer3,name='layer3_n_')
#                self.l3_hist=tf.summary.histogram('l3_c_n_',layer3_n)
#                self.summary_set.append(self.l3_hist)
                
#            layer1 = tf.layers.dense(net, H1, activation=tf.nn.relu,
#                    kernel_initializer=init_w, bias_initializer=init_b, name='l2',
#                    trainable=trainable)
#            layer2 = tf.layers.dense(layer1, H1, activation=tf.nn.relu,
#                    kernel_initializer=init_w, bias_initializer=init_b, name='l3',
#                    trainable=trainable)
            
#            layer2 = tf.layers.dense(layer1, H2, activation=tf.nn.relu,
#                    kernel_initializer=init_w, bias_initializer=init_b, name='l3',
#                    trainable=trainable)
            
            with tf.variable_scope('q'):

                wq_s = tf.get_variable('wq_s', [n_l3, 1], initializer=init_w, trainable=trainable)
#                bq = tf.get_variable('bq', [1, 1], initializer=init_b, trainable=trainable)
#                q = tf.matmul(layer3_n, wq_s)
                q = tf.matmul(layer3, wq_s)
#                self.q_hist=tf.summary.histogram('q_hist_',q)
#                self.summary_set.append(self.q_hist)
#                self.q_w_hist=tf.summary.histogram('q_c_w_',wq_s)
#                self.summary_set.append(self.q_w_hist)
                
#                self.q_b_hist=tf.summary.histogram('q_c_b_',bq)
#                self.summary_set.append(self.q_b_hist)  
                
#                q = tf.layers.dense(layer3,1,kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
#                q = tf.layers.dense(layer3,1,activation=tf.nn.tanh,kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
                q_mean,q_var=tf.nn.moments(q,0)
                
#                self.q_scalar=tf.summary.scalar('q_scalar_', tf.reshape(q_mean,[]))
#                self.summary_set.append(self.q_scalar)  
                
                
                q_tg=tf.nn.tanh(q*1)
                
            with tf.variable_scope('q_scale'):
                q_scale=tf.matmul(q_tg, [[1.]]) 
        
        return q,q_scale

    def learn(self, s, a, s_, r0, r1, r2, ep_total):       

        self.sess.run(self.soft_replace)
        self.sess.run(self.train_op,feed_dict={S: s, self.a: a , S_: s_, R0: r0, R1: r1, R2: r2})

#        if self.t_replace_counter == self.t_replace_iter:
#            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
#            self.t_replace_counter = 0
#        self.t_replace_counter += 1
        self.loss_step+=1
        self.model_localization.append(ep_total)
        
    def Merge_Summary_End(self, s, a, s_ ,r0, r1, r2):
        
        try:
            self.merge_summary = tf.summary.merge(self.summary_set)
            self.summary_critic = self.sess.run(self.merge_summary,feed_dict={S: s, self.a: a , S_: s_, R0: r0, R1: r1, R2: r2})
            self.writer.add_summary(self.summary_critic,self.loss_step)  
        except:
            print("nothing is here")
    
#    def get_rank(self,s, a, r, s_):
#        
#        rank_q,rank_TD=self.sess.run([self.q,self.loss], feed_dict={S: s, self.a: a, R: r, S_: s_})
#
#        return rank_TD
#    
#    def get_rank_probability(self,reward,max_v):
#        
#        rank_reward_correction=reward+(0-np.exp(max_v)*5/2)
#        beta=3/(np.exp(max_v)*5/2)#95%,5%
#        probability_reward=np.exp(beta*rank_reward_correction)/(1+np.exp(beta*rank_reward_correction))
#        
#        rank_TD_correction=rank_TD+(0-(self.rank_TD_min+(self.rank_TD_max-self.rank_TD_min)/2))
#        beta=3/((self.rank_TD_max-self.rank_TD_min)/2)#99%,1%
#        probability_TD=np.exp(beta*rank_TD_correction)/(1+np.exp(beta*rank_TD_correction))
#        
#        return probability_q,probability_TD
#        return probability_reward
        
class Memory(object):
    
    def __init__(self, capacity,capacity_bound, dims):
        self.capacity = capacity
        self.capacity_bound = capacity_bound
        self.dim=dims
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.epsilon=1e-16
        self.grow_trigger =False
        
    def store_transition(self, s, a, a_i, s_, r1, r2, r3):
        
#        s=(s-np.mean(s))/np.sqrt(np.std(s)**2+self.epsilon)
#        s_=(s_-np.mean(s_))/np.sqrt(np.std(s_)**2+self.epsilon)
        
        transition = np.hstack((s, a, a_i, s_, [r1], [r2], [r3]))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1
        
        if index==self.capacity-1:
            
            self.grow_trigger=True
            
        if self.capacity<self.capacity_bound and self.grow_trigger==True:
#            print("self.capacitybefore:",self.capacity)
            self.capacity+= 1
#            print("self.capacityafter:",self.capacity)
#            print("self.databefore:",self.data)
            self.data=np.append(self.data,np.zeros((1,self.dim)),axis=0)
#            print("self.dataafter:",self.data)
#        elif self.capacity>=self.capacity_bound:

#            print("self.capacity:",self.capacity)
    def sample(self, n):
        
#        idx = critic.TD_step % critic.c_TD
#        critic.TD_set[idx]=rank_TD
#        critic.rank_TD_max=max(critic.TD_set)
#        critic.TD_set_no_zero=filter(lambda x: x !=0,critic.TD_set)
#        critic.TD_set_no_zero= [i for i in critic.TD_set_no_zero]
#        critic.rank_TD_min=min(critic.TD_set_no_zero)
#        critic.TD_step+=1

#        indices=[]
#        while len(indices)<n:
#            
#            idx=np.random.choice(self.capacity)
##            rank_q_temp=np.reshape(self.data[idx][-2:-1],())
#            p_temp_TD=np.reshape(self.data[idx, -2:-1],())
##            print("p_temp_TD:",p_temp_TD)
#            choice_TD=np.random.choice(range(2),p=[p_temp_TD,1-p_temp_TD])
#            if choice_TD==0:
#                
#                indices.append(idx)
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]
    
    def read(self,idx,punish_batch_size):
        
        idxs = np.zeros(punish_batch_size) 
        i=0
        for t in range(idx-punish_batch_size,idx):
            
            idxs[i]=t
            i=i+1
        idxs=idxs.astype(int)
        return self.data[idxs, :]
    
    def write(self,idx,punish_batch_size,punished_reward):
        
        idxs = np.zeros(punish_batch_size) 
        i=0
        for t in range(idx-punish_batch_size,idx):
            idxs[i]=t
            i=i+1
        idxs=idxs.astype(int)
        
        self.data[idxs, -input_dim - 1]=punished_reward
    