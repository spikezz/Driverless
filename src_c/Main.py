#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:23:00 2019

@author: spikezz
"""

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
from std_msgs.msg import Float32MultiArray

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

#rate = rospy.Rate(125) # 10hz
rate = rospy.Rate(20) # 10hz
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

list_yellow_cone_random=client.simGetObjectPosesWithTag("yCone")  
list_blue_cone_random=client.simGetObjectPosesWithTag("bCone") 

while list_yellow_cone_random==[] or list_blue_cone_random==[]:
    
    list_yellow_cone_random=client.simGetObjectPosesWithTag("yCone")  
    list_blue_cone_random=client.simGetObjectPosesWithTag("bCone") 

list_blue_cone=[]
list_yellow_cone=[]
list_blue_cone_curve=[]
list_yellow_cone_curve=[]

for i in range(0,len(list_yellow_cone_random)):
    
    blue_cone=client.simGetObjectPosesWithTag('blue_'+str(i))
    yellow_cone=client.simGetObjectPosesWithTag('yellow_'+str(i))
    
    while blue_cone==[] or yellow_cone==[]:
        
        blue_cone=client.simGetObjectPosesWithTag('blue_'+str(i))
        yellow_cone=client.simGetObjectPosesWithTag('yellow_'+str(i))

    list_blue_cone.append(blue_cone[0])
    list_yellow_cone.append(yellow_cone[0])
    
    blue_cone_copy=copy.deepcopy(blue_cone[0])
    yellow_cone_copy=copy.deepcopy(yellow_cone[0])
    
    blue_cone_copy.position.y_val=(-1)*blue_cone[0].position.y_val
    yellow_cone_copy.position.y_val=(-1)*yellow_cone[0].position.y_val
    list_blue_cone_curve.append(blue_cone_copy)
    list_yellow_cone_curve.append(yellow_cone_copy)

coneback=client.simGetObjectPosesWithTag("finish")

sen_sim=sensor.Sensor_Simulator()

real_lidar=False
image_output=False
plote_animation=False
plote_signal=False
first_person=True

sess = tf.Session()

agent_i=Agent.Agent_Imitation(sess,action_dim,input_dim,action_bound,lr_i,replace_iter_a)
#agent_r=Agent.Agent_Reinforcement(sess,action_dim,input_dim,action_bound,lr_a,lr_c,reward_decay,\
#                                  replace_iter_a,replace_iter_c,C_TD=0)
agent=agent_i
memory_imitation = RL.Memory(memory_capacity,memory_capacity_bound, dims=2 * input_dim + 2*action_dim + 1 + 2)

lidar_asynchron_frame=False
change_pid=0
state_switch=0
predict_step_angle=3
predict_step=24
predict_sample_curverature=8
curverature_flatten_grad=16
cur_min=1
cur_max=0
#camera_msg_list=[None]*200
#print(camera_msg_list)
#kp~(0.03,1.0),ki~(0.06,0.3),kd:(0.0102,0.4)
set_point_speed=20

pid_speed=controller.PID_Controller(0.072, 0.06, 0.006, setpoint=set_point_speed,output_limits=(-1, 1))
open_close_control_rate=0.5

#kp~(0.1,3.6),ki~(0.3,0.3),kd:(0.069,0.4)
set_point_steering=0

pid_steering=controller.PID_Controller(0.54, 0.00,0.72, setpoint=set_point_steering,output_limits=(-1, 1))

action_controller=[0,0]

all_var=True
imitation_learn=False

saver = tools.Saver(sess,LOAD,agent_i.actor,None,all_var,agent_i=True)
Summary=tools.Summary()
Summary_Scope=tools.Summary.Summary_Scope(plote_action=False,plote_speed=False,plote_cross_position=True,plote_predict_angle=False,\
                                    plote_predict_curverature=False)

agent_i.actor.writer.add_graph(sess.graph,episode_counter)

if plote_animation:
    
    lidar_ploter=tools.Ploter([-30,30],[-30,30],[6.4, 4.8])
    curve_ploter=tools.Ploter([-110,90],[-30,170],[12.8, 9.6])

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
    ros_car_state_message=ri.ros_car_state_message_creater(rospy,client,initial_velocoty_noise,time_step,image=image_output)
    
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
      
        list_cone_sensored_sita=copy.deepcopy(list_yellow_cone_sensored_sita)
        list_cone_sensored_sita=list_cone_sensored_sita+list_blue_cone_sensored_sita
       
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
       
        closest_yellow_curve_point_pair,predict_yellow_curve_point_pair,predict_yellow_curverature_point=\
        sen_sim.find_closest_curve_point_pair(yellow_cone_spline_origin,ros_car_state_message[7],predict_step,predict_sample_curverature,predict_step_angle)
        
        distance_between_closet_yellow_curve_point,vector_closest_yellow_curve_point=\
        sen_sim.calculate_distance_curve_point_pair(closest_yellow_curve_point_pair)
        
        distance_between_predict_yellow_curve_point,vector_predict_yellow_curve_point=\
        sen_sim.calculate_distance_curve_point_pair(predict_yellow_curve_point_pair)
        
        closest_blue_curve_point_pair,predict_blue_curve_point_pair,predict_blue_curverature_point=\
        sen_sim.find_closest_curve_point_pair(blue_cone_spline_origin,ros_car_state_message[7],predict_step,predict_sample_curverature,predict_step_angle)
        
        distance_between_closet_blue_curve_point,vector_closest_blue_curve_point=\
        sen_sim.calculate_distance_curve_point_pair(closest_blue_curve_point_pair)
        
        distance_between_predict_blue_curve_point,vector_predict_blue_curve_point=\
        sen_sim.calculate_distance_curve_point_pair(predict_blue_curve_point_pair)    
        
        predict_angle_difference=sen_sim.calculate_predict_angle_difference(predict_yellow_curve_point_pair,ros_car_state_message[8])
        
        if distance_between_predict_yellow_curve_point<distance_between_closet_blue_curve_point:
            
            curverature_sign=-1
            
        elif distance_between_predict_yellow_curve_point>=distance_between_closet_blue_curve_point:
            
            curverature_sign=1
            
        
        try:
            
            sin_projection_yellow=sen_sim.calculate_sinus_projection_closest_curve_point_pair(closest_yellow_curve_point_pair,\
                                                                                              distance_between_closet_yellow_curve_point)-2.5
            sin_projection_blue=sen_sim.calculate_sinus_projection_closest_curve_point_pair(closest_blue_curve_point_pair,\
                                                                                            distance_between_closet_blue_curve_point)
            sin_projection_difference=sin_projection_yellow-sin_projection_blue
            
            if sin_projection_yellow>5:
                
                sin_projection_blue=sin_projection_blue*(-1)
                
            if sin_projection_blue>5:
                
                sin_projection_yellow=sin_projection_yellow*(-1)
                
#            print("sin_projection_yellow:%.2f \t sin_projection_blue:%.2f"%(sin_projection_yellow,sin_projection_blue))
#            print("sin_projection difference:%.2f"%(sin_projection_difference))
        except:
            
            print("shit happens")
            
        predict_yellow_curverature_point_c=copy.deepcopy(predict_yellow_curverature_point)
        predict_blue_curverature_point_c=copy.deepcopy(predict_blue_curverature_point)

        if first_person:
            
            sen_sim.rotate_sight(list_cone_sensored_sita,closest_yellow_curve_point_pair,closest_blue_curve_point_pair,\
                                 predict_yellow_curve_point_pair,predict_blue_curve_point_pair,predict_yellow_curverature_point_c,\
                                 predict_blue_curverature_point_c,ros_car_state_message[8])  
        
        x_center_yellow,y_center_yellow,r_yellow,circle_norm_vector_yellow=cal.calculate_circle_curverature_with_3p([predict_yellow_curverature_point_c[0][1][1],\
                                                                    predict_yellow_curverature_point_c[1][1][1],predict_yellow_curverature_point_c[2][1][1]],\
                                                                    [predict_yellow_curverature_point_c[0][1][0],predict_yellow_curverature_point_c[1][1][0],\
                                                                     predict_yellow_curverature_point_c[2][1][0]])
        
        x_center_blue,y_center_blue,r_blue,circle_norm_vector_blue=cal.calculate_circle_curverature_with_3p([predict_blue_curverature_point_c[0][1][1],\
                                                                    predict_blue_curverature_point_c[1][1][1],predict_blue_curverature_point_c[2][1][1]],\
                                                                    [predict_blue_curverature_point_c[0][1][0],predict_blue_curverature_point_c[1][1][0],\
                                                                     predict_blue_curverature_point_c[2][1][0]])
                                                                    
        agent.actor.predict_curverature_measured.append((1/r_yellow+1/r_blue)/2)
        
        if len(agent.actor.predict_curverature_measured)>(predict_sample_curverature+curverature_flatten_grad):
            
            sum_flattened_predict_curverature=0
            sum_flattened_predict_curverature_gradient=0
            
            for i in range(0,predict_sample_curverature+curverature_flatten_grad):
       
                sum_flattened_predict_curverature=agent.actor.predict_curverature_measured[-(i+1)]+sum_flattened_predict_curverature
                
                if i<(predict_sample_curverature+curverature_flatten_grad-1):
                    
                    sum_flattened_predict_curverature_gradient=np.abs(agent.actor.predict_curverature_measured[-(i+1)]-\
                    agent.actor.predict_curverature_measured[-(i+2)])+sum_flattened_predict_curverature_gradient
         
            flattened_predict_curverature=sum_flattened_predict_curverature/(predict_sample_curverature+curverature_flatten_grad*(1+sum_flattened_predict_curverature_gradient)**1)
            
            if flattened_predict_curverature<cur_min:
                
                cur_min=flattened_predict_curverature
                
            if flattened_predict_curverature>cur_max:
                
                cur_max=flattened_predict_curverature
            
            agent.actor.predict_curverature.append(flattened_predict_curverature)
            
#        print('cur_min:%.10f cur_max:%.10f'%(cur_min,cur_max))
            
        if plote_animation:

            x_rs,y_rs = cal.draw_circle(x_center_yellow,y_center_yellow,r_yellow,lidar_ploter)
            sen_sim.plot_all(list_cone_sensored_sita,closest_yellow_curve_point_pair,closest_blue_curve_point_pair,\
                             predict_yellow_curve_point_pair,predict_blue_curve_point_pair,predict_yellow_curverature_point_c,\
                             predict_blue_curverature_point_c,x_rs,y_rs,lidar_ploter,ros_car_state_message[8],first_person)
#        
##        #chose action imitation
##        action = agent.actor.choose_action(observation,agent_i=True)
##            
##        if action[0]>=0.5:
##            
##            if imitation_learn:
##                
##                car_controls.throttle=float(action[0]-0.5)
##                car_controls.brake=0
##                
##            agent.actor.throttle_imitation.append(car_controls.throttle)  
##            agent.actor.brake_imitation.append(car_controls.brake)
##            
##        elif action[0]<=-0.5:
##            
##            if imitation_learn:
##                
##                car_controls.brake=float(-action[0]-0.5)
##                car_controls.throttle=0
##                
##            agent.actor.throttle_imitation.append(car_controls.throttle) 
##            agent.actor.brake_imitation.append(car_controls.brake)
##            
##        else:
##            
##            agent.actor.throttle_imitation.append(action[0])  
##            
##        if imitation_learn:    
##            
##            car_controls.steering=float(action[1])
##            
##        agent.actor.steering_angle_imitation.append(action[1])
          
        #controller behavior
#        speed=np.sqrt(np.power(ros_car_state_message[0].x, 2)+np.power(ros_car_state_message[1].x, 2)+\
#                np.power(ros_car_state_message[2].x, 2))
        
#        print("speed:",speed)
#        print("speed error:",car_state.speed-speed)

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
#        if elapsed_time_setpoint>=10: 
            
#            car_controls.steering=(elapsed_time_setpoint-10)*0.01
#            car_controls.steering=np.sin((elapsed_time_setpoint-10)/2)*0.1
#        
#        if elapsed_time_setpoint>=10 and elapsed_time_setpoint<22:
#            set_point_speed=set_point_speed-(elapsed_time_setpoint-10)*0.05
#            pid_speed.setpoint=set_point_speed
#            
#        if elapsed_time_setpoint>=22:
#            
#            if set_point_speed<20:
#                
#                set_point_speed=set_point_speed+(elapsed_time_setpoint-22)*0.05
#                pid_speed.setpoint=set_point_speed
    
            
            

#        try:
#            
##            set_point_speed=(-1000)*(flattened_predict_curverature-np.abs(flattened_predict_curverature-agent.actor.predict_curverature[-2]))+20
#            set_point_speed=math.exp(5.9-flattened_predict_curverature)-349
##            print(set_point_speed)
#
#            if set_point_speed<15:
#                
#                set_point_speed=15
##            
##            open_close_control_rate=(-24)*flattened_predict_curverature+0.9
#            
#            if open_close_control_rate<0:
#                
#                open_close_control_rate=0
#                
#        except:
#            
#            print('not ready!')
            
        agent.actor.speed.append(car_state.speed)
        agent.actor.set_point_speed.append(set_point_speed)
        agent.actor.lateral_position.append(sin_projection_yellow)
        agent.actor.set_point_lateral_position.append(set_point_steering)
        agent.actor.predict_angle_diffrence.append(-math.radians(predict_angle_difference))
        agent.actor.set_point_predict_angle_diffrence.append(0)
        agent.actor.open_close_control_rate.append(open_close_control_rate)
        
#        print('sin_projection_yellow:',sin_projection_yellow-1.5)
        pid_speed.setpoint=set_point_speed
        action_controller[0]=pid_speed(car_state.speed)
#        action_controller[0]=1
#        if elapsed_time_setpoint>=10: 
#            
#            action_controller[0]=0
            
        action_controller[1]=pid_steering(sin_projection_yellow)/((set_point_speed)**1)
        
#        car_controls.steering=predict_angle_difference/40*open_close_control_rate+action_controller[1]*(1-open_close_control_rate)
#        car_controls.steering=predict_angle_difference/40*0.5+action_controller[1]*0.1
        car_controls.steering=action_controller[1]*0.5
        
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
            
        act_msg=Float32MultiArray()
        now=rospy.get_rostime()
        act_msg.data.append(now.secs)
        act_msg.data.append(now.nsecs)
        act_msg.data.append(car_controls.throttle)
        act_msg.data.append(car_controls.brake)
        act_msg.data.append(car_controls.steering)
        
        client.setCarControls(car_controls)   
        
        agent.actor.odm_msg.append(ros_car_state_message[7])
        agent.actor.eul_msg.append(ros_car_state_message[8])
        
        if time_step%4==0:
            
            Summary_Scope.plot_summary(agent_i)
        
#        elapsed_time_setpoint=time.time()-time_stamp_setpoint
        time_set.append(elapsed_time_setpoint)
#        print("elapsed_time:",elapsed_time_setpoint)
        
#        if elapsed_time_setpoint>=20:
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
        
#        if lidar_frame:
#            
#            ros_publisher['pub_lidar_data'].publish(pointcloud_msg)
#            lidar_frame=not lidar_frame
#            
#        else:
#            
#            lidar_frame=not lidar_frame
#            
        ros_publisher['pub_Odometry_auto'].publish(ros_car_state_message[7])
        
    if image_output:
        
        try:
            
            ros_publisher['pub_Image'].publish(ros_car_state_message[9])
#            camera_msg_list[time_step%100]=ros_car_state_message[9]
            
        except:
            
            print('leer')
            
    ros_publisher['pub_action'].publish(act_msg)
    rate.sleep()
    elapsed_time=time.time()-time_stamp
    time_step +=1
    time_step_set.append(elapsed_time)
    print("elapsed_time:",elapsed_time)
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