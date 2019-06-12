#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:38:44 2019

@author: spikezz
"""

import Neurocore
import copy
import math

import numpy as np
import tensorflow as tf

from simple_pid import PID

class Agent_Controller(object):
    
    def __init__(self,action_dim,kp_speed,ki_speed,kd_speed,limits_speed,set_point_speed_min,\
                 kp_steering,ki_steering,kd_steering,set_point_steering,limits_steering):
        
        ##dimension of action
        self.action_dim = 2
        self.action=np.zeros(self.action_dim)
        self.action_old=np.zeros(self.action_dim)
        self.state_idx=0
        self.recorder_switch=0
        self.predict_step_angle=3
        self.predict_step=16
        self.curverature_sample_step=8
        self.open_control_rate=0.4
        self.close_control_rate=1-self.open_control_rate
        self.set_point_speed=set_point_speed_min
        self.set_point_speed_step=0.2
        self.set_point_speed_min=set_point_speed_min
        self.set_point_steering=set_point_steering
        self.kp_speed=kp_speed
        self.ki_speed=ki_speed
        self.kd_speed=kd_speed
        self.limits_speed=limits_speed
        self.kp_steering=kp_steering
        self.ki_steering=ki_steering
        self.kd_steering=kd_steering
        self.limits_steering=limits_steering
        self.speed_controller=PID(Kp=kp_speed, Ki=ki_speed, Kd=kd_speed,setpoint=self.set_point_speed,output_limits=limits_speed)
        self.steering_controller=PID(Kp=kp_steering, Ki=ki_steering, Kd=kd_steering,setpoint=self.set_point_steering,output_limits=limits_steering)
    
    def update_setpoint_speed(self,curverature_meter,agent_imitation,sensor):
        
        if hasattr(curverature_meter, 'flattened_predict_curverature'):

            self.set_point_speed_end=math.exp(5.9-curverature_meter.flattened_predict_curverature)-349
            
            if np.abs(self.set_point_speed-self.set_point_speed_end)<=self.set_point_speed_step or (self.set_point_speed>self.set_point_speed_end):
                
                self.set_point_speed=self.set_point_speed_end
                
            elif self.set_point_speed<self.set_point_speed_end:
                
                self.set_point_speed=self.set_point_speed+self.set_point_speed_step
            
            if self.set_point_speed<self.set_point_speed_min:
                
                self.set_point_speed=self.set_point_speed_min 
            
            self.speed_controller.setpoint=self.set_point_speed
            agent_imitation.speed.append(sensor.car_state.speed)
            agent_imitation.set_point_speed.append(self.set_point_speed)
            
    def reset(self,client,car_controls):
        
        car_controls.throttle=0
        car_controls.brake=1
        car_controls.steering=0
        client.setCarControls(car_controls) 
        self.set_point_speed=self.set_point_speed_min
        self.speed_controller=PID(Kp=self.kp_speed, Ki=self.ki_speed, Kd=self.kd_speed,setpoint=self.set_point_speed,output_limits=self.limits_speed)
        self.steering_controller=PID(Kp=self.kp_steering, Ki=self.ki_steering, Kd=self.kd_steering,setpoint=self.set_point_steering,output_limits=self.limits_steering)
        
class Agent_Imitation(object):
    
    def __init__(self,input_dim,action_dim,sess):
        
        #session
        self.sess=sess
        #dimension of input data
        self.input_dim = input_dim
        #dimension of action
        self.action_dim = action_dim
        #action limits
        self.action_boundary=np.array([1.0,1.0])
        # imitation learning rate for actor
        self.lr_i = 6.25e-6
        #initial memory capacity
        self.memory_capacity = 512
        #memory capacity upper bound
        self.memory_capacity_boundary = 8192
        #size of memory slice
        self.memory_batch = 16
        #load model
        self.load=False
        #action of Agent
        self.action=np.zeros(self.action_dim)
        #copy of action
        self.action_old=np.zeros(self.action_dim)
        #inputs state of Agent
        self.observation=np.zeros(self.input_dim)
        #copy the state
        self.observation_old=np.zeros(self.input_dim)
        #tensorboard data recorder
        self.writer = tf.summary.FileWriter("/home/spikezz/KaratUnreal/Clean Code/Driverless/src/logs")
        self.summary_set=[]
        #define neuro actor
        self.actor=Neurocore.Actor_Imitation(self,input_dim,action_dim,sess,self.action_boundary,self.lr_i)
        #data history:odm
        self.odm_msg=[]
        self.eul_msg=[]
        #data history:steering
        self.steering_angle_controller=[]
        self.steering_angle_imitation=[]
        self.diff_steering=[]
        #data history:throttle
        self.throttle_controller=[]
        self.throttle_imitation=[]
        self.diff_throttle=[]
        #data history:brake
        self.brake_controller=[]
        self.brake_imitation=[]
        self.diff_brake=[]
        #data history:speed
        self.speed=[]
        self.set_point_speed=[]
        self.optic_speed=[]
        #data history:lateral position
        self.lateral_position=[]
        self.set_point_lateral_position=[]
        #data history:speed
        self.predict_angle_diffrence=[]
        #data history:curverature
        self.predict_curverature_measured=[]
        self.predict_curverature=[]
        #data history:mix controller
        self.open_control_rate=[]
        #data history:time
        self.time_step_set_episode=[]
        self.time_step_set=[]
        #create memory
        self.M=self.Memory(self.memory_capacity,self.memory_capacity_boundary,memory_dimension=input_dim + 2*action_dim)
        
    def update_observation(self,sensor,sensor_visualizer,car_controls):
        
        observation_temp=[[],[],[],[],[],[],[]]
        
        for i in range(0,len(sensor_visualizer.list_sensored_cone_covered_free)):
            
            input_cone_norm=[sensor_visualizer.list_sensored_cone_covered_free[i][2][0]/5,sensor_visualizer.list_sensored_cone_covered_free[i][2][1]/5]
            #yellow cone
            if sensor_visualizer.list_sensored_cone_covered_free[i][3]=='y':
                
                observation_temp[0].append(input_cone_norm)
            #blue cone
            elif sensor_visualizer.list_sensored_cone_covered_free[i][3]=='b':
                
                observation_temp[1].append(input_cone_norm)
                
            else:
                
                print('cone cover function defekt!!!')
            
        if len(observation_temp[0])!=0:
            
            observation_temp[0]=np.vstack(observation_temp[0]).ravel()
            
        if len(observation_temp[1])!=0:
            
            observation_temp[1]=np.vstack(observation_temp[1]).ravel()
        
        observation_temp[2]=np.vstack([sensor.car_state.speed/5]).ravel()
        observation_temp[3]=np.vstack([sensor.velocity_2d_correction[1][0]/5,sensor.velocity_2d_correction[1][1]/5]).ravel()
        observation_temp[4]=np.vstack([sensor.car_state_message[2].z/(2*math.pi)*36]).ravel()
        observation_temp[5]=np.vstack([sensor.car_state_message[3].x/5,sensor.car_state_message[3].y/5]).ravel()
        observation_temp[6]=np.vstack([car_controls.steering*3]).ravel()
        
        observation_temp_pack=np.hstack((observation_temp[2],observation_temp[3],observation_temp[4],observation_temp[5],observation_temp[6]))  
#        print('0',observation_temp[0])
#        print('1',observation_temp[1])
#        print('2',observation_temp[2])
#        print('3',observation_temp[3])
#        print('4',observation_temp[4])
#        print('5',observation_temp[5])
#        print('entire',observation_temp_pack)
        for t in range(0,len(observation_temp[0])):
            
            self.observation[t]=observation_temp[0][t]
        
        for t in range(20,20+len(observation_temp[1])):
            
            self.observation[t]=observation_temp[1][t-20]
        
        for t in range(40,40+len(observation_temp_pack)):
            
            self.observation[t]=observation_temp_pack[t-40]
            
    def sample_and_learn(self):
        
        if self.M.pointer>self.M.initial_capacity:
        
            memory_slice = self.M.sample(self.memory_batch)
            state_slice = memory_slice[:, :self.input_dim]
#            agent_i_action_slice = memory_slice[:, -2*self.action_dim:-self.action_dim]
            agent_c_action_slice = memory_slice[:, -self.action_dim:]  
    
            self.actor.learn_Imitation(state_slice,agent_c_action_slice)
            
    def reset(self,client):
        
        client.reset()
    
        #data history:odm
        self.odm_msg=[]
        self.eul_msg=[]
        #data history:steering
        self.steering_angle_controller=[]
        self.steering_angle_imitation=[]
        self.diff_steering=[]
        #data history:throttle
        self.throttle_controller=[]
        self.throttle_imitation=[]
        self.diff_throttle=[]
        #data history:brake
        self.brake_controller=[]
        self.brake_imitation=[]
        self.diff_brake=[]
        #data history:speed
        self.speed=[]
        self.set_point_speed=[]
        self.optic_speed=[]
        #data history:lateral position
        self.lateral_position=[]
        self.set_point_lateral_position=[]
        #data history:speed
        self.predict_angle_diffrence=[]
        #data history:curverature
        self.predict_curverature_measured=[]
        self.predict_curverature=[]
        #data history:mix controller
        self.open_control_rate=[]
        #data history:time
        self.time_step_set_episode=[]
        self.time_step_set=[]
        
    def merge_summary(self,actor,s,a_c,time_step):  
        
#        try:
            
        self.merge = tf.summary.merge(self.summary_set)
        self.summary_agent=self.sess.run(self.merge,feed_dict={actor.S: s, actor.A_C:a_c})
        self.writer.add_summary(self.summary_agent,time_step)

#        except:
#            
#            print("nothing is here")

    class Memory(object):
    
        def __init__(self, capacity,capacity_boundary, memory_dimension):
            
            self.capacity = capacity
            self.initial_capacity=copy.deepcopy(self.capacity)
            self.capacity_boundary = capacity_boundary
            self.dimension=memory_dimension
            self.data = np.zeros((capacity, self.dimension))
            self.pointer = 0
            self.epsilon=1e-16
            self.grow_trigger =False
            
        def store_transition(self, s, a_i, a_c):
  
            transition = np.hstack((s, a_i, a_c))
            index = self.pointer % self.capacity  # replace the old memory with new memory
            self.data[index, :] = transition
            self.pointer += 1
            
            if index==self.capacity-1:
                
                self.grow_trigger=True
                
            if self.capacity<self.capacity_boundary and self.grow_trigger==True:
    
                self.capacity+= 1
                self.data=np.append(self.data,np.zeros((1,self.dimension)),axis=0)
                
        def sample(self, n):
            
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
        
        def write(self,idx,punish_batch_size,punished_reward,agent):
            
            idxs = np.zeros(punish_batch_size) 
            
            i=0
            for t in range(idx-punish_batch_size,idx):
                idxs[i]=t
                i=i+1
            idxs=idxs.astype(int)
            
            self.data[idxs, -agent.input_dim - 1]=punished_reward
        
            
            
            
            
            
