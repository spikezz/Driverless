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
#import math
import copy

import ROS_Interface as RI
#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

time.sleep(3)

remote_control=False
plote_animation=True
plote_scope=True
model_data_recording=False
episode_counter=0
summary=True

collision_distance=0.3
collision=False

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
car_state_initial = client.getCarState()

initializer=Initialization.Initializer(car_state_initial)
initial_velocoty_noise=initializer.initial_velocoty_noise

ros_interface=RI.ROS_Interface(client,car_controls,remote_control)
ros_publisher=ros_interface.ros_publisher

if remote_control:
    
    ros_subscriber=ros_interface.ros_subscriber

rate = rospy.Rate(10)#*2

lidar=Sensor.Sensor_Box.Real_Lidar(client)
sensor=Sensor.Sensor_Box(client,ros_interface,initializer,lidar,episode_counter)
sv=Sensor.Sensor_Box.Sensor_Visualizer(draw_sight=plote_animation)


sess = tf.Session()

learning_phase=0

if learning_phase==0:
    
    agent_c=Agent.Agent_Controller(action_dim=2,kp_speed=0.072,ki_speed=0.06,kd_speed=0.006,limits_speed=(-1, 1),set_point_speed_min=5,\
                                   kp_steering=0.54,ki_steering=0,kd_steering=0.72,set_point_steering=0,limits_steering=(-1, 1))
    agent_i=Agent.Agent_Imitation(input_dim=50,action_dim=2,sess=sess)
    agent_i.writer.add_graph(sess.graph,episode_counter)
    imitation_drive=False
    distance=0
    speed_measurement=0
    
elif learning_phase==1:
    
    pass

cone_set=Enviroment.Cone_set(client,initializer,ros_interface,sv,agent_c,draw_map=plote_animation,loop_circuit=True)

sess.run(tf.global_variables_initializer())

cm=Sensor.Sensor_Box.Curverature_Meter()
scope=Tools.Summary_Scope(plot_action=plote_scope,plot_speed=plote_scope,plote_cross_position=plote_scope,plot_time=plote_scope)

elapsed_time_entire_old=0
porsition_old=[0,0,0]

time_stamp_entire= time.time()

while not rospy.is_shutdown():
    time_stamp_cycle = time.time()
    sensor.update(client,ros_interface,initializer,lidar,agent_i,episode_counter,create_image_message=False,create_lidar_message=False)
    
    if not remote_control:
        
        if not summary:
         
            if plote_animation:
                
                cone_set.global_map.update(sensor.car_state_message[5])
                sv.lidar_top_view.clear()
                
            sv.update(cone_set,sensor,agent_c,agent_i,cm,draw_sight=plote_animation,first_person=True)
            
            if learning_phase==0:
                
                agent_i.action=agent_i.actor.choose_action(agent_i.observation_old)
                
                if agent_i.action[0]>=0:
                    
                    if imitation_drive:
                            
                        car_controls.throttle=float(agent_i.action[0])
                        car_controls.brake=0  
                        
                    agent_i.throttle_imitation.append(float(agent_i.action[0]))  
                    agent_i.brake_imitation.append(0)
                    
                elif agent_i.action[0]<=-0:
                     
                    if imitation_drive:
            
                        car_controls.brake=float(-agent_i.action[0])
                        car_controls.throttle=0
                        
                    agent_i.brake_imitation.append(float(-agent_i.action[0])) 
                    agent_i.throttle_imitation.append(0)
                    
                if imitation_drive:
                    
                    car_controls.steering=float(agent_i.action[1])
                    
                agent_i.steering_angle_imitation.append(agent_i.action[1])
                
                agent_c.update_setpoint_speed(cm,agent_i,sensor)
                agent_c.action[0]=agent_c.speed_controller(sensor.car_state.speed)
                
                if agent_c.action[0]>=0:
                    
                    if not imitation_drive:
                        
                        car_controls.throttle=float(agent_c.action[0])
                        car_controls.brake=0
                        
                    agent_i.throttle_controller.append(float(agent_c.action[0]))
                    agent_i.brake_controller.append(0)
                    
                else:
                    
                    if not imitation_drive:
                        
                        car_controls.brake=float(-agent_c.action[0])
                        car_controls.throttle=0
                        
                    agent_i.throttle_controller.append(float(-agent_c.action[0]))
                    agent_i.brake_controller.append(0)
                
                agent_c.action[1]=sv.predict_angle_difference/34*agent_c.open_control_rate+\
                (agent_c.steering_controller(sv.mittle_position)/((agent_c.set_point_speed)**1))*agent_c.close_control_rate
                
                if not imitation_drive:

                    car_controls.steering=agent_c.action[1]
                
                agent_i.steering_angle_controller.append(agent_c.action[1])
                
                agent_i.action_0_controller.append(agent_c.action[0])
                agent_i.action_0_imitation.append(agent_i.action[0])
                agent_i.diff_action_0.append(agent_c.action[0]-agent_i.action[0])
                agent_i.action_1_controller.append(agent_c.action[1])
                agent_i.action_1_imitation.append(agent_i.action[1])
                agent_i.diff_action_1.append(agent_c.action[1]-agent_i.action[1])
                
                client.setCarControls(car_controls) 
                    
                    
                action_message=ros_interface.create_action_message(car_controls)
                
                if model_data_recording:
                    
                    agent_i.odm_msg.append(sensor.car_state_message[5])
                    agent_i.eul_msg.append(sensor.car_state_message[6])
                    
                agent_i.update_observation(sensor,sv,car_controls)
         
        #        print('agent_i.observation',agent_i.observation)
                
                if episode_counter >0:
                
                    agent_i.M.store_transition(agent_i.observation_old,agent_i.action,agent_c.action)
                
                if sv.sin_projection_yellow<collision_distance or sv.sin_projection_blue<collision_distance:
                    print('collision!!!')
                    collision=True
                    summary=True
                    
                if (sv.closest_yellow_curve_point_pair[0][2] > cone_set.start_point-6 and sv.closest_yellow_curve_point_pair[0][2] < cone_set.start_point-3 )\
                    or (sv.closest_yellow_curve_point_pair[1][2] > cone_set.start_point-6 and sv.closest_yellow_curve_point_pair[1][2] < cone_set.start_point-3 ):
                    
                    print('now:%.2f \t %.2f \t setpoint:%.2f'%(sv.closest_yellow_curve_point_pair[0][2],sv.closest_yellow_curve_point_pair[1][2],cone_set.start_point))
                    print('reset')
                    summary=True
                
                agent_i.sample_and_learn()
                
            elif learning_phase==1:
                
                pass
            
            if episode_counter%128==0 and plote_scope:
                
                scope.plot_summary(agent_i) 
                agent_i.merge_summary(agent_i.actor,agent_i.observation_old[np.newaxis, :],agent_c.action[np.newaxis, :],episode_counter)
                
            agent_i.observation_old=copy.deepcopy(agent_i.observation)
          
        else:
            
            collision=False
            summary=False
            agent_i.reset(client)
            agent_c.reset(client,car_controls)
            time.sleep(3)
            time_stamp_entire= time.time()
            print('episode_counter:',episode_counter)
            
            if episode_counter>agent_i.memory_capacity_boundary :
                
                imitation_drive=True
    else:
        
        if episode_counter%16==0:
            
            scope.plot_summary(agent_i) 
            
#        print(sensor.car_state_message[5]) 
#        print(sensor.velocity_2d_correction) 
#        sensor.car_state_message[5].twist.twist.linear.x=sensor.velocity_2d_correction[1][0]
#        sensor.car_state_message[5].twist.twist.linear.y=sensor.velocity_2d_correction[1][1]
#        print(sensor.car_state_message[5]) 
#        ros_publisher['pub_blue_cone'].publish(cone_set.blue_cone_message)
#        ros_publisher['pub_yellow_cone'].publish(cone_set.yellow_cone_message)
#        ros_publisher['pub_odometry_auto'].publish(sensor.car_state_message[5])
        
    rate.sleep()
    elapsed_time_entire=time.time()-time_stamp_entire
    agent_i.time_step_set_episode.append(elapsed_time_entire)
    elapsed_time_cycle=time.time()-time_stamp_cycle
    agent_i.time_step_set.append(elapsed_time_cycle)
 
   
#    print('elapsed_time_entire',elapsed_time_entire)
#    print('elapsed_time_cycle',elapsed_time_cycle)
    episode_counter+=1