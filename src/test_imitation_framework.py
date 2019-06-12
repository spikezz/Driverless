#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 23:53:17 2019

@author: spikezz
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
import airsim
import rospy 
import Initialization
import Enviroment
import Agent
import Sensor
import Tools
import math
import copy

import ROS_Interface as RI
#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

time.sleep(3)

plote_animation=True
plote_scope=True
model_data_recording=False
episode_counter=0
summary=False

collision_distance=0.3
collision=False

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
car_state_initial = client.getCarState()

initializer=Initialization.Initializer(car_state_initial)
initial_velocoty_noise=initializer.initial_velocoty_noise

ros_interface=RI.ROS_Interface()
ros_publisher=ros_interface.ros_publisher

rate = rospy.Rate(20) 

lidar=Sensor.Sensor_Box.Real_Lidar(client)
sensor=Sensor.Sensor_Box(client,ros_interface,initializer,lidar,episode_counter)
sv=Sensor.Sensor_Box.Sensor_Visualizer(draw_sight=plote_animation)


sess = tf.Session()

learning_phase=0

if learning_phase==0:
    
    agent_c=Agent.Agent_Controller(action_dim=2,kp_speed=0.072,ki_speed=0.06,kd_speed=0.006,limits_speed=(-1, 1),set_point_speed_min=3,\
                                   kp_steering=0.9,ki_steering=0,kd_steering=1.78,set_point_steering=2.5,limits_steering=(-1, 1))
    agent_i=Agent.Agent_Imitation(input_dim=50,action_dim=2,sess=sess)
    agent_i.writer.add_graph(sess.graph,episode_counter)
    imitation_drive=False
    
elif learning_phase==1:
    
    pass

cone_set=Enviroment.Cone_set(client,initializer,ros_interface,sv,agent_c,draw_map=plote_animation,loop_circuit=True)

sess.run(tf.global_variables_initializer())

cm=Sensor.Sensor_Box.Curverature_Meter()
scope=Tools.Summary_Scope(plot_action=plote_scope,plot_speed=plote_scope)

time_stamp_entire= time.time()

log_i_old=0

for i in range (10000000):
    
    action_i=agent_i.actor.choose_action(agent_i.observation_old)
            
    agent_i.observation=np.zeros(agent_i.input_dim)
    
    if i%4==0:
        
        for j in range(0,round(0.25*len(agent_i.observation))):
            
#            agent_i.observation[j]=-1.29
            agent_i.observation[j]=0
            
        for j in range(round(0.25*len(agent_i.observation)),round(0.5*len(agent_i.observation))):
            
#            agent_i.observation[j]=1.29
            agent_i.observation[j]=30
            
        for j in range(round(0.5*len(agent_i.observation)),round(0.75*len(agent_i.observation))):
            
#            agent_i.observation[j]=0.43
            agent_i.observation[j]=20
            
        for j in range(round(0.75*len(agent_i.observation)),round(len(agent_i.observation))):
            
#            agent_i.observation[j]=-0.43
            agent_i.observation[j]=10
            
    elif i%4==1:
        
        for j in range(0,round(0.25*len(agent_i.observation))):
            
#            agent_i.observation[j]=-0.43
            agent_i.observation[j]=10
            
        for j in range(round(0.25*len(agent_i.observation)),round(0.5*len(agent_i.observation))):
            
#            agent_i.observation[j]=-1.29
            agent_i.observation[j]=0
        
        for j in range(round(0.5*len(agent_i.observation)),round(0.75*len(agent_i.observation))):
            
#            agent_i.observation[j]=1.29
            agent_i.observation[j]=30
            
        for j in range(round(0.75*len(agent_i.observation)),round(len(agent_i.observation))):
            
#            agent_i.observation[j]=0.43
            agent_i.observation[j]=20
    
    elif i%4==2:
        
        for j in range(0,round(0.25*len(agent_i.observation))):
            
#            agent_i.observation[j]=0.43
            agent_i.observation[j]=20
 
        for j in range(round(0.25*len(agent_i.observation)),round(0.5*len(agent_i.observation))):
            
#            agent_i.observation[j]=-0.43
            agent_i.observation[j]=10
        
        for j in range(round(0.5*len(agent_i.observation)),round(0.75*len(agent_i.observation))):
            
#            agent_i.observation[j]=-1.29
            agent_i.observation[j]=0
            
        for j in range(round(0.75*len(agent_i.observation)),round(len(agent_i.observation))):
            
#            agent_i.observation[j]=1.29
            agent_i.observation[j]=30
        
    elif i%4==3:
        
        for j in range(0,round(0.25*len(agent_i.observation))):
            
#            agent_i.observation[j]=1.29
            agent_i.observation[j]=30
 
        for j in range(round(0.25*len(agent_i.observation)),round(0.5*len(agent_i.observation))):
            
#            agent_i.observation[j]=0.43
            agent_i.observation[j]=20
        
        for j in range(round(0.5*len(agent_i.observation)),round(0.75*len(agent_i.observation))):
            
#            agent_i.observation[j]=-0.43
            agent_i.observation[j]=10
            
        for j in range(round(0.75*len(agent_i.observation)),round(len(agent_i.observation))):
            
#            agent_i.observation[j]=1.29
            agent_i.observation[j]=30
            
    action_c=np.array([1/((i-1)%4+1),1/((i-1)%4+1)])
    
    if i >0:
        
        agent_i.M.store_transition(agent_i.observation_old,action_i,action_c)
        
    agent_i.throttle_controller.append(action_c[0])
    agent_i.throttle_imitation.append(action_i[0])
    agent_i.diff_throttle.append(action_c[0]-action_i[0])
    agent_i.steering_angle_controller.append(action_c[1])
    agent_i.steering_angle_imitation.append(action_i[1])
    agent_i.diff_steering.append(action_c[1]-action_i[1])

    if agent_i.M.pointer>agent_i.M.initial_capacity:
        
        memory_slice = agent_i.M.sample(agent_i.memory_batch)
        state_slice = memory_slice[:, :agent_i.input_dim]
        agent_i_action_slice = memory_slice[:, -2*agent_i.action_dim:-agent_i.action_dim]
        agent_c_action_slice = memory_slice[:, -agent_i.action_dim:]  

#        print('memory_slice:',memory_slice)
#        print('state_slice:',state_slice)
#        print('agent_i_action_slice:',agent_i_action_slice)
#        print('agent_c_action_slice:',agent_c_action_slice)
#        print('i_grads:',agent_i.actor.i_grads)
#        print('i_grads length:',len(agent_i.actor.i_grads))
#        print('e_params:',agent_i.actor.e_params)
#        print('e_params length:',len(agent_i.actor.e_params))
        
        agent_i.actor.learn_Imitation(state_slice,agent_c_action_slice)
        
        if i%128==0 or i%128==1 or i%128==2 or i%128==3:
            
#            print(agent_i.observation)
            print('action_c:',action_c)
            print('action_i:',action_i)
            print([action_c[0]-action_i[0],action_c[1]-action_i[1]],agent_i.M.pointer)
            print('log10(i):', math.floor(np.log10(i))) 
            
            agent_i.throttle_controller=[]
            agent_i.throttle_imitation=[]
            agent_i.diff_throttle=[]
            agent_i.steering_angle_controller=[]
            agent_i.steering_angle_imitation=[]
            agent_i.diff_steering=[]
        
        agent_i.merge_summary(agent_i.actor,agent_i.observation_old[np.newaxis, :],action_c[np.newaxis, :],i)
            
        if i%4==0:
        
            scope.plot_summary(agent_i) 
            
    agent_i.observation_old=copy.deepcopy(agent_i.observation)
    
#    if i==5000 :
#        
#        agent_i.actor.lr_i=agent_i.actor.lr_i*0.1
#        print('lr:',agent_i.actor.lr_i)
#        log_i_old=math.floor(np.log10(i))
##    if i==35000 : 
##        
##        agent_i.actor.lr_i=agent_i.actor.lr_i*0.1
##        print('lr:',agent_i.actor.lr_i)
#        
#    if i>5000 :    
#        
#        if  math.floor(np.log10(i))>log_i_old:
#            
#            agent_i.actor.lr_i=agent_i.actor.lr_i*0.1
#            print('lr:',agent_i.actor.lr_i)
#            log_i_old=math.floor(np.log10(i))
                    
print('End!!!')