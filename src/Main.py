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

time.sleep(1)

#switches
model_data_recording=False
pid_tuning=False
plote_animation=True
plote_scope=False
remote_control=False
sensor_noise=True
summary=False
summerize_loss=True
training=0

collision_distance=0.7
collision=False

episode_counter=0
elapsed_time_entire_old=0
porsition_old=[0,0,0]
elapsed_time_entire=0

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

rate = rospy.Rate(10)#*2.1

sess = tf.Session()

learning_phase=0

if learning_phase==0:
    
    agent_c=Agent.Agent_Controller(action_dim=2,kp_speed=0.45,ki_speed=0.00773,kd_speed=0.0766,limits_speed=(-1, 1),set_point_speed_min=4,set_point_speed_max=12,\
                                   kp_steering=0.69,ki_steering=0,kd_steering=1.92,set_point_steering=0,limits_steering=(-1, 1),pid_tuning=pid_tuning)
    agent_i=Agent.Agent_Imitation(input_dim=20,action_dim=2,sess=sess,training_phase=training)
    agent_i.writer.add_graph(sess.graph,episode_counter)
    imitation_drive=True
    distance=0
    speed_measurement=0
    agent_i.reset(client)
    agent_c.reset(client,car_controls)
    
elif learning_phase==1:
    
    pass

lidar=Sensor.Sensor_Box.Real_Lidar(client)
sensor=Sensor.Sensor_Box(client,ros_interface,initializer,lidar,episode_counter)
sv=Sensor.Sensor_Box.Sensor_Visualizer(draw_sight=plote_animation)

cone_set=Enviroment.Cone_set(client,initializer,ros_interface,sv,agent_c,draw_map=plote_animation,loop_circuit=True)

sess.run(tf.global_variables_initializer())

cm=Sensor.Sensor_Box.Curverature_Meter()
scope=Tools.Summary_Scope(plot_action=True,plot_speed=True,plote_cross_position=True,plot_time=plote_scope,\
                          plot_synchronization=plote_scope)

sensor.update(client,ros_interface,initializer,lidar,agent_i,episode_counter,create_image_message=False,create_lidar_message=False)
sv.initialze_vector_root(cone_set,sensor)

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
                
                agent_i.imitate(car_controls,imitation_drive=imitation_drive)
                agent_c.drive(car_controls,agent_i,cm,sensor, sv,elapsed_time_entire,imitation_drive=imitation_drive,pid_tuning=pid_tuning)
                
                client.setCarControls(car_controls) 
                  
                action_message=ros_interface.create_action_message(car_controls)
                
                if model_data_recording:
                    
                    agent_i.odm_msg.append(sensor.car_state_message[5])
                    agent_i.eul_msg.append(sensor.car_state_message[6])
                    
                agent_i.update_observation(sensor,sv,car_controls,sensor_noise)
         
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
            
            if episode_counter%64==0:
                
                scope.plot_summary(agent_i) 
            
            if episode_counter%256==0 and summerize_loss:
                
                agent_i.merge_summary(agent_i.actor,agent_i.observation_old[np.newaxis, :],agent_c.action[np.newaxis, :],episode_counter)
                
            agent_i.observation_old=copy.deepcopy(agent_i.observation)
          
        else:
            
            collision=False
            summary=False
            agent_i.reset(client)
            agent_c.reset(client,car_controls)
            time.sleep(1)
            time_stamp_entire= time.time()
            print('cur_min:%.10f cur_max:%.10f'%(cm.curv_min,cm.curv_max))
            print('episode_counter:',episode_counter)
            sensor.update(client,ros_interface,initializer,lidar,agent_i,episode_counter,create_image_message=False,create_lidar_message=False)
            sv.initialze_vector_root(cone_set,sensor)
            
            if episode_counter>agent_i.memory_capacity_boundary :
                
                imitation_drive=True
                
    else:
        
        pass
#        if episode_counter%16==0:
#            
#            scope.plot_summary(agent_i) 
            
#        ros_publisher['pub_odometry_auto'].publish(sensor.car_state_message[5])
        
    rate.sleep()
    elapsed_time_entire=time.time()-time_stamp_entire
    agent_i.time_step_set_episode.append(elapsed_time_entire)
    elapsed_time_cycle=time.time()-time_stamp_cycle
    agent_i.time_step_set.append(elapsed_time_cycle)
 
    time_diff=elapsed_time_cycle
    distance=np.sqrt(np.power((sensor.car_state_message[5].pose.pose.position.x-porsition_old[0]), 2)+\
                     np.power((sensor.car_state_message[5].pose.pose.position.y-porsition_old[1]), 2)+\
                     np.power((sensor.car_state_message[5].pose.pose.position.z-porsition_old[2]), 2))
    speed_measurement=distance/(time_diff+1e-16)
    
    if speed_measurement>30:
        
        speed_measurement=0
        
    speed_rate=sensor.car_state.speed/(speed_measurement+1e-16)
    
    if speed_rate>10:
        
        speed_rate=0
        
    agent_i.speed_measurement.append(speed_measurement)
    agent_i.speed_rate.append(speed_rate)
    porsition_old[0]=copy.deepcopy(sensor.car_state_message[5].pose.pose.position.x)
    porsition_old[1]=copy.deepcopy(sensor.car_state_message[5].pose.pose.position.y)
    porsition_old[2]=copy.deepcopy(sensor.car_state_message[5].pose.pose.position.z)
    elapsed_time_entire_old=copy.deepcopy(elapsed_time_entire)
    episode_counter+=1