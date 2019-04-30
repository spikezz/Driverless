#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:49:30 2019

@author: spikezz
"""

import RL


class Agent_Imitation(object):
    
    def __init__(self,sess,action_dimension,input_dimension,action_bound,lr_i,replace_iter_a=None):

        self.actor = RL.Actor(sess, action_dimension, input_dimension, action_bound,lr_i = lr_i ,agent_i=True)
        
class Agent_Reinforcement(object):
    
    def __init__(self,sess,action_dimension,input_dimension,action_bound,lr_a,lr_c,reward_decay,replace_iter_a,replace_iter_c,C_TD):
        
        self.actor = RL.Actor(sess, action_dimension, input_dimension, action_bound,lr_r = lr_a, agent_r=True)
        
        self.critic = RL.Critic(sess, action_dimension, input_dimension, lr_c, reward_decay, self.actor.a, self.actor.a_,replace_iter_c, C_TD)
        
        self.actor.add_grad_to_graph(self.critic.a_grads)