#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:23:00 2019

@author: spikezz
"""
#from airsim import ImageRequest
#from geometry_msgs.msg import Vector3
#from geometry_msgs.msg import Quaternion
#from geometry_msgs.msg import PoseArray
#from geometry_msgs.msg import Pose
#from nav_msgs.msg import Odometry
#from std_msgs.msg import Float32MultiArray
#from std_msgs.msg import Float64
#from scipy.special import softmax
#import PythonTrackWrapper as ptw

import numpy as np
import calculate as cal
import tensorflow as tf
import main_function as mf
import ROS_Interface as ri
import matplotlib.pyplot as plt
import airsim
import tools
import RL
import time
import rospy 
import Agent 
import sensor
import copy

time.sleep(2)

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
car_controls = airsim.CarControls()

car_state = client.getCarState()

initial_velocoty_noise=mf.Calibration_Speed_Sensor(car_state)

ros_publisher=ri.define_ros_publisher(rospy)

rate = rospy.Rate(60) # 10hz

##dimension of input data
input_dim = 100
##dimension of input data
##dimension of action
action_dim = 2
##dimension of action
action_bound=np.array([0.5,1])
#action_bound = np.array([-0.5,0.5],[-0.5,0.5],[-1,1])
# learning rate for actor
LR_A = 1.25e-8
# learning rate for actor
# imitation learning rate for actor
lr_i = 1.25e-4
# imitation learning rate for actor
# learning rate for critic
LR_C = 6.25e-8
# learning rate for critic
# reward discount
rd = 0.9
# reward discount
#after this learning number of main net update the target net of actor
replace_iter_a = 1024
#after this learning number of main net update the target net of actor
##after this learning number of main net update the target net of Critic
replace_iter_c = 1024
##after this learning number of main net update the target net of Critic

LOAD=False
summary=False
episode_counter=0

memory_capacity = 256
memory_capacity_bound = 65536

actionorigin=np.zeros(action_dim)
action_old=np.zeros(action_dim)
actionorigin_old=np.zeros(action_dim)

#inputs state of RL Agent
observation=np.zeros(input_dim)
#inputs state of RL Agentprojection
#copy the state
observation_old=np.zeros(input_dim)

#list_blue_cone_random=client.simGetObjectPoses("RightCone")
list_yellow_cone_random=client.simGetObjectPoses("yCone")  
list_blue_cone_random=client.simGetObjectPoses("bCone") 

while list_yellow_cone_random==[] or list_blue_cone_random==[]:
    
    list_yellow_cone_random=client.simGetObjectPoses("yCone")  
    list_blue_cone_random=client.simGetObjectPoses("bCone") 
#blue=client.simGetObjectPoses("blue_5")
#yellow=client.simGetObjectPoses("yellow_5") 
#print("blue_5:",blue)
#print("yellow_5:",yellow) 
list_blue_cone=[]
list_yellow_cone=[]
list_blue_cone_curve=[]
list_yellow_cone_curve=[]

print("lengthy:",len(list_yellow_cone_random))
#print("list_yellow_cone_random:",list_yellow_cone_random)
print("lengthb:",len(list_blue_cone_random))
#print("list_blue_cone_random:",list_blue_cone_random)

for i in range(0,len(list_yellow_cone_random)):
    
    blue_cone=client.simGetObjectPoses('blue_'+str(i))
    yellow_cone=client.simGetObjectPoses('yellow_'+str(i))
    #print('blue_'+str(i))
    #print('yellow_'+str(i))
    #print("blue_cone:",blue_cone[0])
    #print("yellow_cone:",yellow_cone[0])
    
    while blue_cone==[] or yellow_cone==[]:
        
        blue_cone=client.simGetObjectPoses('blue_'+str(i))
        yellow_cone=client.simGetObjectPoses('yellow_'+str(i))
#        print('blue_'+str(i))
#        print('yellow_'+str(i))
#        print("blue_cone:",blue_cone[0])
#        print("yellow_cone:",yellow_cone[0])
        
#    print("i:",i)
    list_blue_cone.append(blue_cone[0])
    list_yellow_cone.append(yellow_cone[0])
    
    blue_cone_copy=copy.deepcopy(blue_cone[0])
    yellow_cone_copy=copy.deepcopy(yellow_cone[0])
    
    blue_cone_copy.position.y_val=(-1)*blue_cone[0].position.y_val
    yellow_cone_copy.position.y_val=(-1)*yellow_cone[0].position.y_val
    list_blue_cone_curve.append(blue_cone_copy)
    list_yellow_cone_curve.append(yellow_cone_copy)
#    print("blue_cone",blue_cone)
    

#print("list_blue_cone",list_blue_cone)
#print("list_yellow_cone",list_yellow_cone)
coneback=client.simGetObjectPoses("finish")

sen_sim=sensor.Sensor_Simulator()

sess = tf.Session()

agent_i=Agent.Agent_Imitation(sess,action_dim,input_dim,action_bound,lr_i,replace_iter_a)
#agent_r=ag.Agent_Reinforcement(sess,action_dim,input_dim,action_bound,lr_i,replace_iter_a)
memory_imitation = RL.Memory(memory_capacity,memory_capacity_bound, dims=2 * input_dim + 2*action_dim + 1 + 2)

all_var=True
reinforcement=False

saver = tools.Saver(sess,LOAD,agent_i.actor,None,all_var,reinforcement)

agent_i.actor.writer.add_graph(sess.graph,episode_counter)

lidar_ploter=tools.Ploter([-15,15],[-15,15],[6.4, 4.8])
curve_ploter=tools.Ploter([-80,30],[-80,30],[12.8, 9.6])

cone_message=ri.ros_cone_message_creater(rospy,list_blue_cone,list_yellow_cone)

blue_cone_spline=sen_sim.draw_cone_spline(list_blue_cone_curve,curve_ploter,'b',draw=True)
yellow_cone_spline=sen_sim.draw_cone_spline(list_yellow_cone_curve,curve_ploter,'y',draw=True)

blue_cone_spline_origin=sen_sim.draw_cone_spline(list_blue_cone,curve_ploter,'b',draw=False)
yellow_cone_spline_origin=sen_sim.draw_cone_spline(list_yellow_cone,curve_ploter,'y',draw=False)

print("length_idx_blue",len(blue_cone_spline[0]))
print("length_idx_yellow",len(yellow_cone_spline[0]))

lidar=sensor.Lidar_Real(client)

#main loop
while not rospy.is_shutdown():
    
#    car_controls.throttle=0.1
#    car_controls.steering=0.0
#    client.setCarControls(car_controls)
    
    time_stamp = time.time()
    car_state = client.getCarState()
    ros_car_state_message=ri.ros_car_state_message_creater(rospy,car_state,initial_velocoty_noise)
    pointcloud_msg=lidar.get_lidar_data(ros_car_state_message[7])
    if summary==False:

        if len(curve_ploter.ax.lines)>4:
            
            curve_ploter.ax.lines.pop(-1)
            
        lidar_ploter.ax.lines.clear()
        sen_sim.update_position_cone_spline(ros_car_state_message[7],curve_ploter)
        
#        list_yellow_cone_sensored_distance,list_yellow_cone_sensored_sita=\
#        sen_sim.find_sensored_cone(list_yellow_cone,ros_car_state_message[4],ros_car_state_message[5],ploter,tools.plt,color='y')
#        list_blue_cone_sensored_distance,list_blue_cone_sensored_sita=\
#        sen_sim.find_sensored_cone(list_blue_cone,ros_car_state_message[4],ros_car_state_message[5],ploter,tools.plt,color='b')
        
        list_yellow_cone_sensored_sita=\
        sen_sim.find_sensored_cone(list_yellow_cone,ros_car_state_message[7],ros_car_state_message[8],color='y')
        list_blue_cone_sensored_sita=\
        sen_sim.find_sensored_cone(list_blue_cone,ros_car_state_message[7],ros_car_state_message[8],color='b')
        
#        list_cone_sensored_distance=copy.copy(list_yellow_cone_sensored_distance)
#        list_cone_sensored_distance=list_cone_sensored_distance+list_blue_cone_sensored_distance
        
        list_cone_sensored_sita=copy.deepcopy(list_yellow_cone_sensored_sita)
        list_cone_sensored_sita=list_cone_sensored_sita+list_blue_cone_sensored_sita
        
#        list_cone_sensored_distance=sorted(list_cone_sensored_distance,key=lambda x:x[0])
        list_cone_sensored_sita=sorted(list_cone_sensored_sita,key=lambda x:x[1])

        list_sensored_cone_yellow_covered_free,list_sensored_cone_blue_covered_free=sen_sim.cover_cone(list_cone_sensored_sita,lidar_ploter)
        
        if len(list_sensored_cone_yellow_covered_free)>=2:
            
            closest_yellow_cone_pair=sen_sim.find_closest_cone_pair(list_sensored_cone_yellow_covered_free)
            distance_between_yellow_cone,vector_closest_yellow_cone=sen_sim.calculate_distance_closest_cone_pair(closest_yellow_cone_pair)
            
        else:

            if len(list_sensored_cone_yellow_covered_free)==0:
                
                print("no yellow cone")
                
            else:
                
                closest_yellow_cone=list_sensored_cone_yellow_covered_free
                
                print("single yellow cone")
                
        if len(list_sensored_cone_blue_covered_free)>=2:
            
            closest_blue_cone_pair=sen_sim.find_closest_cone_pair(list_sensored_cone_blue_covered_free)
            distance_between_blue_cone,vector_closest_blue_yellow_cone=sen_sim.calculate_distance_closest_cone_pair(closest_blue_cone_pair)
            
        else:

            if len(list_sensored_cone_blue_covered_free)==0:
                
                print("no blue cone")
                
            else:
                
                closest_blue_cone=list_sensored_cone_blue_covered_free
                
                print("single blue cone")
                
#        print("distance_between_yellow_cone",distance_between_yellow_cone)
#        print("vector_closest_yellow_cone",vector_closest_yellow_cone)
#        print("distance_between_blue_cone",distance_between_blue_cone)
#        print("vector_closest_blue_yellow_cone",vector_closest_blue_yellow_cone)
#        print("closest_yellow_cone_pair",closest_yellow_cone_pair)
#        print("closest_blue_cone_pair",closest_blue_cone_pair)
                
        closest_yellow_curve_point_pair=sen_sim.find_closest_curve_point_pair(yellow_cone_spline_origin,ros_car_state_message[7])
        distance_between_yellow_curve_point,vector_closest_yellow_curve_point=sen_sim.calculate_distance_closest_curve_point_pair(closest_yellow_curve_point_pair)
        
        closest_blue_curve_point_pair=sen_sim.find_closest_curve_point_pair(blue_cone_spline_origin,ros_car_state_message[7])
        distance_between_blue_curve_point,vector_closest_blue_curve_point=sen_sim.calculate_distance_closest_curve_point_pair(closest_blue_curve_point_pair)
        
        try:
            
            sin_projection_yellow=sen_sim.calculate_sinus_projection_closest_curve_point_pair(closest_yellow_curve_point_pair,distance_between_yellow_curve_point)
            sin_projection_blue=sen_sim.calculate_sinus_projection_closest_curve_point_pair(closest_blue_curve_point_pair,distance_between_blue_curve_point)
            print("sin_projection_yellow:%.2f \t sin_projection_blue:%.2f"%(sin_projection_yellow,sin_projection_blue))
            
        except:
            
            print("shit happens")
            
        sen_sim.plot_all(list_cone_sensored_sita,closest_yellow_curve_point_pair,closest_blue_curve_point_pair,lidar_ploter,ros_car_state_message[8])
        
#        print("sin_projection_yellow:%.2f \t sin_projection_blue:%.2f"%(sin_projection_yellow,sin_projection_blue))

#        try:
#                
#            closest_yellow_cone_pair=sen_sim.find_closest_cone_pair(list_sensored_cone_yellow_covered_free)
#            closest_blue_cone_pair=sen_sim.find_closest_cone_pair(list_sensored_cone_blue_covered_free)
#            
#            distance_between_yellow_cone,vector_closest_cone=sen_sim.calculate_distance_closest_cone_pair(closest_yellow_cone_pair)
#            distance_between_blue_cone,vector_closest_cone=sen_sim.calculate_distance_closest_cone_pair(closest_blue_cone_pair)
#            
#        except:
#            try:
#                
#            print("shit happens")
        

            

            
#    ros_publisher['pub_euler'].publish(ros_car_state_message[5])
#    ros_publisher['pub_blue_cone'].publish(bcn_msg)
#    ros_publisher['pub_yellow_cone'].publish(ycn_msg)
    ros_publisher['pub_lidar_data'].publish(pointcloud_msg)
#    ros_publisher['pub_Twist'].publish(ros_car_state_message[3])
#    ros_publisher['pub_Quaternion'].publish(ros_car_state_message[4])
    rate.sleep()
    