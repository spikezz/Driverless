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
import controller
import math

time.sleep(3)
time_step=0
cycle_time=0.2
time_set=[]
time_step_set=[]

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

car_state = client.getCarState()

initial_velocoty_noise=mf.Calibration_Speed_Sensor(car_state)

ros_publisher=ri.define_ros_publisher(rospy)

rate = rospy.Rate(10) # 10hz

##dimension of input data
input_dim = 100
##dimension of input data
##dimension of action
action_dim = 2
##dimension of action
action_bound=np.array([1.5,1])
#action_bound = np.array([-0.5,0.5],[-0.5,0.5],[-1,1])
# learning rate for actor
lr_a = 1.25e-5
# learning rate for actor
# imitation learning rate for actor
lr_i = 1.25e-4
# imitation learning rate for actor
# learning rate for critic
lr_c = 6.25e-5
# learning rate for critic
# reward discount
reward_decay = 0.9
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

list_yellow_cone_random=client.simGetObjectPoses("yCone")  
list_blue_cone_random=client.simGetObjectPoses("bCone") 

while list_yellow_cone_random==[] or list_blue_cone_random==[]:
    
    list_yellow_cone_random=client.simGetObjectPoses("yCone")  
    list_blue_cone_random=client.simGetObjectPoses("bCone") 

list_blue_cone=[]
list_yellow_cone=[]
list_blue_cone_curve=[]
list_yellow_cone_curve=[]

print("lengthy:",len(list_yellow_cone_random))
print("lengthb:",len(list_blue_cone_random))

for i in range(0,len(list_yellow_cone_random)):
    
    blue_cone=client.simGetObjectPoses('blue_'+str(i))
    yellow_cone=client.simGetObjectPoses('yellow_'+str(i))
    
    while blue_cone==[] or yellow_cone==[]:
        
        blue_cone=client.simGetObjectPoses('blue_'+str(i))
        yellow_cone=client.simGetObjectPoses('yellow_'+str(i))

    list_blue_cone.append(blue_cone[0])
    list_yellow_cone.append(yellow_cone[0])
    
    blue_cone_copy=copy.deepcopy(blue_cone[0])
    yellow_cone_copy=copy.deepcopy(yellow_cone[0])
    
    blue_cone_copy.position.y_val=(-1)*blue_cone[0].position.y_val
    yellow_cone_copy.position.y_val=(-1)*yellow_cone[0].position.y_val
    list_blue_cone_curve.append(blue_cone_copy)
    list_yellow_cone_curve.append(yellow_cone_copy)

coneback=client.simGetObjectPoses("finish")

sen_sim=sensor.Sensor_Simulator()

real_lidar=False
image_output=False
plote_animation=True
first_person=True

sess = tf.Session()

agent_i=Agent.Agent_Imitation(sess,action_dim,input_dim,action_bound,lr_i,replace_iter_a)
#agent_r=Agent.Agent_Reinforcement(sess,action_dim,input_dim,action_bound,lr_a,lr_c,reward_decay,\
#                                  replace_iter_a,replace_iter_c,C_TD=0)
agent=agent_i
memory_imitation = RL.Memory(memory_capacity,memory_capacity_bound, dims=2 * input_dim + 2*action_dim + 1 + 2)

change_pid=0
state_switch=0
predict_step=5

#kp~(0.03,1.0),ki~(0.06,0.3),kd:(0.0102,0.4)
set_point_speed=20
pid_speed=controller.PID_Controller(0.048, 0.06, 0.0102, setpoint=set_point_speed,output_limits=(-1, 1))

open_close_control_rate=0.5
#kp~(0.1,0.9),ki~(0.3,0.3),kd:(0.069,0.4)
set_point_steering=1.5
pid_steering=controller.PID_Controller(0.3, 0.045,0.6 , setpoint=set_point_steering,output_limits=(-1, 1))
action_controller=[0,0]
steering_open=0

all_var=True
imitation_learn=False

saver = tools.Saver(sess,LOAD,agent_i.actor,None,all_var,agent_i=True)
Summary=tools.Summary()

agent_i.actor.writer.add_graph(sess.graph,episode_counter)

if plote_animation:
    
    lidar_ploter=tools.Ploter([-15,15],[-15,15],[6.4, 4.8])
    curve_ploter=tools.Ploter([-20,90],[-80,30],[12.8, 9.6])

cone_message=ri.ros_cone_message_creater(rospy,list_blue_cone,list_yellow_cone)

if plote_animation:
    
    blue_cone_spline=sen_sim.draw_cone_spline(list_blue_cone_curve,'b',curve_ploter,draw=True)
    yellow_cone_spline=sen_sim.draw_cone_spline(list_yellow_cone_curve,'y',curve_ploter,draw=True)

    
else:
    
    blue_cone_spline=sen_sim.draw_cone_spline(list_blue_cone_curve,'b',ploter=None,draw=False)
    yellow_cone_spline=sen_sim.draw_cone_spline(list_yellow_cone_curve,'y',ploter=None,draw=False)
    
blue_cone_spline_origin=sen_sim.draw_cone_spline(list_blue_cone,'b',ploter=None,draw=False)
yellow_cone_spline_origin=sen_sim.draw_cone_spline(list_yellow_cone,'y',ploter=None,draw=False)
    

print("length_idx_blue",len(blue_cone_spline[0]))
print("length_idx_yellow",len(yellow_cone_spline[0]))

if real_lidar:
    
    lidar=sensor.Lidar_Real(client)
    
time_stamp_setpoint = time.time()
#main loop
while not rospy.is_shutdown():
    
    time_stamp = time.time()
    
    car_state = client.getCarState()
    ros_car_state_message=ri.ros_car_state_message_creater(rospy,client,initial_velocoty_noise,image=image_output)
    
    if real_lidar:
        
        pointcloud_msg=lidar.get_lidar_data(ros_car_state_message[7])
        
    if summary==False:
        
        if plote_animation:
            
            if len(curve_ploter.ax.lines)>4:
                
                curve_ploter.ax.lines.pop(-1)
                
            lidar_ploter.ax.lines.clear()
            
            sen_sim.update_position_cone_spline(ros_car_state_message[7],curve_ploter)

        list_yellow_cone_sensored_sita=\
        sen_sim.find_sensored_cone(list_yellow_cone,ros_car_state_message[7],ros_car_state_message[8],color='y')
        list_blue_cone_sensored_sita=\
        sen_sim.find_sensored_cone(list_blue_cone,ros_car_state_message[7],ros_car_state_message[8],color='b')
        
##        list_cone_sensored_distance=copy.copy(list_yellow_cone_sensored_distance)
##        list_cone_sensored_distance=list_cone_sensored_distance+list_blue_cone_sensored_distance
        
        list_cone_sensored_sita=copy.deepcopy(list_yellow_cone_sensored_sita)
        list_cone_sensored_sita=list_cone_sensored_sita+list_blue_cone_sensored_sita
        
##        list_cone_sensored_distance=sorted(list_cone_sensored_distance,key=lambda x:x[0])
        
        list_cone_sensored_sita=sorted(list_cone_sensored_sita,key=lambda x:x[1])

        list_sensored_cone_yellow_covered_free,list_sensored_cone_blue_covered_free=sen_sim.cover_cone(list_cone_sensored_sita)
        
        if len(list_sensored_cone_yellow_covered_free)>=2:
            
            closest_yellow_cone_pair=sen_sim.find_closest_cone_pair(list_sensored_cone_yellow_covered_free)
            distance_between_yellow_cone,vector_closest_yellow_cone=sen_sim.calculate_distance_closest_cone_pair(closest_yellow_cone_pair)
            
        else:

            if len(list_sensored_cone_yellow_covered_free)==1:

                closest_yellow_cone=list_sensored_cone_yellow_covered_free
                print("single yellow cone")
                
            else:
                
                print("no yellow cone")
                
        if len(list_sensored_cone_blue_covered_free)>=2:
            
            closest_blue_cone_pair=sen_sim.find_closest_cone_pair(list_sensored_cone_blue_covered_free)
            distance_between_blue_cone,vector_closest_blue_yellow_cone=sen_sim.calculate_distance_closest_cone_pair(closest_blue_cone_pair)
            
        else:

            if len(list_sensored_cone_blue_covered_free)==1:
                
                closest_blue_cone=list_sensored_cone_blue_covered_free
                print("single blue cone")
                
            else:
                
                print("no blue cone")
       
        closest_yellow_curve_point_pair,predict_yellow_curve_point_pair=\
        sen_sim.find_closest_curve_point_pair(yellow_cone_spline_origin,ros_car_state_message[7],predict_step)
        
        distance_between_closet_yellow_curve_point,vector_closest_yellow_curve_point=\
        sen_sim.calculate_distance_curve_point_pair(closest_yellow_curve_point_pair)
        
        distance_between_predict_yellow_curve_point,vector_predict_yellow_curve_point=\
        sen_sim.calculate_distance_curve_point_pair(predict_yellow_curve_point_pair)
        
        closest_blue_curve_point_pair,predict_blue_curve_point_pair=\
        sen_sim.find_closest_curve_point_pair(blue_cone_spline_origin,ros_car_state_message[7],predict_step)
        
        distance_between_closet_blue_curve_point,vector_closest_blue_curve_point=\
        sen_sim.calculate_distance_curve_point_pair(closest_blue_curve_point_pair)
        
        distance_between_predict_blue_curve_point,vector_predict_blue_curve_point=\
        sen_sim.calculate_distance_curve_point_pair(predict_blue_curve_point_pair)
        
        if predict_yellow_curve_point_pair[1][2]>predict_yellow_curve_point_pair[0][2]:
            
            predict_angle_difference=cal.calculate_sita_of_radius(predict_yellow_curve_point_pair[0][1],\
                                    predict_yellow_curve_point_pair[1][1])-math.degrees(ros_car_state_message[8].z)
            
        else:
            
            predict_angle_difference=cal.calculate_sita_of_radius(predict_yellow_curve_point_pair[1][1],\
                                    predict_yellow_curve_point_pair[0][1])-math.degrees(ros_car_state_message[8].z)

        if predict_angle_difference > 180:
            
            predict_angle_difference=predict_angle_difference-360
            
        elif predict_angle_difference < -180:
            
            predict_angle_difference=predict_angle_difference+360

        if predict_angle_difference>90:
            
            predict_angle_difference=180-predict_angle_difference
            
        elif predict_angle_difference<-90:
            
            predict_angle_difference=180+predict_angle_difference
            
#        print('predict_angle_difference:',predict_angle_difference)
        
        try:
            
            sin_projection_yellow=sen_sim.calculate_sinus_projection_closest_curve_point_pair(closest_yellow_curve_point_pair,\
                                                                                              distance_between_closet_yellow_curve_point)
            sin_projection_blue=sen_sim.calculate_sinus_projection_closest_curve_point_pair(closest_blue_curve_point_pair,\
                                                                                            distance_between_closet_blue_curve_point)
            sin_projection_difference=sin_projection_yellow-sin_projection_blue
            
            if sin_projection_yellow>3:
                
                sin_projection_blue=sin_projection_blue*(-1)
                
            if sin_projection_blue>3:
                
                sin_projection_yellow=sin_projection_yellow*(-1)
                
            print("sin_projection_yellow:%.2f \t sin_projection_blue:%.2f"%(sin_projection_yellow,sin_projection_blue))
#            print("sin_projection difference:%.2f"%(sin_projection_difference))
        except:
            
            print("shit happens")
            
        if first_person:
            
            sen_sim.rotate_sight(list_cone_sensored_sita,closest_yellow_curve_point_pair,closest_blue_curve_point_pair,\
                                 predict_yellow_curve_point_pair,predict_blue_curve_point_pair,ros_car_state_message[8])
  
        if plote_animation:
            
            sen_sim.plot_all(list_cone_sensored_sita,closest_yellow_curve_point_pair,closest_blue_curve_point_pair,\
                             predict_yellow_curve_point_pair,predict_blue_curve_point_pair,\
                             lidar_ploter,ros_car_state_message[8],first_person)
        
#        #chose action imitation
#        action = agent.actor.choose_action(observation,agent_i=True)
#            
#        if action[0]>=0.5:
#            
#            if imitation_learn:
#                
#                car_controls.throttle=float(action[0]-0.5)
#                car_controls.brake=0
#                
#            agent.actor.throttle_imitation.append(car_controls.throttle)  
#            agent.actor.brake_imitation.append(car_controls.brake)
#            
#        elif action[0]<=-0.5:
#            
#            if imitation_learn:
#                
#                car_controls.brake=float(-action[0]-0.5)
#                car_controls.throttle=0
#                
#            agent.actor.throttle_imitation.append(car_controls.throttle) 
#            agent.actor.brake_imitation.append(car_controls.brake)
#            
#        else:
#            
#            agent.actor.throttle_imitation.append(action[0])  
#            
#        if imitation_learn:    
#            
#            car_controls.steering=float(action[1])
#            
#        agent.actor.steering_angle_imitation.append(action[1])
          
        #controller behavior
#        speed=np.sqrt(np.power(ros_car_state_message[0].x, 2)+np.power(ros_car_state_message[1].x, 2)+\
#                np.power(ros_car_state_message[2].x, 2))
#        
#        print("speed:",speed)
#        print("speed error:",car_state.speed-speed)
#        agent.actor.speed.append(car_state.speed)
#        agent.actor.speed.append(speed)
#
#        if car_state.speed>set_point_speed:
#            
#            print("step:",len(agent.actor.speed))
#        print(car_state.speed)

        elapsed_time_setpoint=time.time()-time_stamp_setpoint
        
#        print(elapsed_time_setpoint)
        
#        if elapsed_time_setpoint>=12 and change_pid==0:
#            set_point_speed=10
#            pid_speed.setpoint=set_point_speed
#            change_pid=1
#            
#        if elapsed_time_setpoint>=20 and change_pid==1:
#            set_point_speed=20
#            pid_speed.setpoint=set_point_speed
#            change_pid=2
#            
#        if elapsed_time_setpoint>=28 and change_pid==2:
#            set_point_speed=15
#            pid_speed.setpoint=set_point_speed
#            change_pid=3
#            
#        if elapsed_time_setpoint>=36 and change_pid==3:
#            set_point_speed=5
#            pid_speed.setpoint=set_point_speed      
#            change_pid=4
#        
#        if elapsed_time_setpoint>=10 and elapsed_time_setpoint<22:
##        if elapsed_time_setpoint>=10: 
##            car_controls.steering=(elapsed_time_setpoint-10)*0.01
##            car_controls.steering=np.sin((elapsed_time_setpoint-10)/2)*1
#            set_point_speed=set_point_speed-(elapsed_time_setpoint-10)*0.03
#            pid_speed.setpoint=set_point_speed
#            
#        if elapsed_time_setpoint>=22:
#            
#            if set_point_speed<20:
#                
#                set_point_speed=set_point_speed+(elapsed_time_setpoint-22)*0.03
#                pid_speed.setpoint=set_point_speed
        
#        if change_pid==0:
#            
#            set_point_steering=sin_projection_yellow
#            change_pid=1
            
        agent.actor.speed.append(car_state.speed)
        agent.actor.set_point_speed.append(set_point_speed)
        agent.actor.lateral_position.append(sin_projection_yellow)
        agent.actor.set_point_lateral_position.append(set_point_steering)
        agent.actor.predict_angle_diffrence.append(-math.radians(predict_angle_difference))
        agent.actor.set_point_predict_angle_diffrence.append(0)
        
        action_controller[0]=pid_speed(car_state.speed)
#        action_controller[0]=1
        
        action_controller[1]=pid_steering(sin_projection_yellow)/(set_point_speed**1)
        car_controls.steering=predict_angle_difference/40*open_close_control_rate+action_controller[1]*(1-open_close_control_rate)
#        car_controls.steering=action_controller[1]
        agent.actor.steering_angle_controller.append(car_controls.steering)
        
        if action_controller[0]>=0:
            
            car_controls.throttle=float(action_controller[0])
            car_controls.brake=0
            agent.actor.throttle_controller.append(car_controls.throttle)
            agent.actor.brake_controller.append(car_controls.brake)
            
        else:
            
            car_controls.brake=float(-action_controller[0])
            car_controls.throttle=0
            agent.actor.throttle_controller.append(car_controls.throttle)
            agent.actor.brake_controller.append(car_controls.brake)
            
        client.setCarControls(car_controls)   
        
        agent.actor.odm_msg.append(ros_car_state_message[7])
        agent.actor.eul_msg.append(ros_car_state_message[8])

#        Summary.plot_summary(agent_i)
        
        elapsed_time_setpoint=time.time()-time_stamp_setpoint
        time_set.append(elapsed_time_setpoint)
#        print("elapsed_time:",elapsed_time_setpoint)
#        
#        if elapsed_time_setpoint>=24:
#
#            summary=True
            
    elif summary==True and state_switch==0:
        
        car_controls.throttle=0
        car_controls.brake=1
        car_controls.steering=0
        client.setCarControls(car_controls)   
        print('write data!')
        Summary.writer_data(agent_i,agent.actor.odm_msg,agent.actor.eul_msg,time_set)
        state_switch=1
        plt.ioff()
        plt.show()
        break
 
    if real_lidar:
        
        ros_publisher['pub_lidar_data'].publish(pointcloud_msg)
        ros_publisher['pub_Odometry_auto'].publish(ros_car_state_message[7])
        
    if image_output:
        try:
            ros_publisher['pub_Image'].publish(ros_car_state_message[9])
#            ros_publisher['pub_Odometry_auto'].publish(ros_car_state_message[7])
        except:
            print('leer')
    rate.sleep()
    elapsed_time=time.time()-time_stamp
    time_step +=1
#    time_step_set.append(elapsed_time)
#    print("elapsed_time:",elapsed_time)
#    if elapsed_time>0.05:
#        print("sth broke:%d"%(time_step))
#        break
        
#    if elapsed_time<cycle_time:
#        
#        time.sleep(cycle_time-elapsed_time) 
#        
#    else:
#        pass
#        print("elapsed_time:",elapsed_time)