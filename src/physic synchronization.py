#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 22:08:17 2019

@author: spikezz
"""

    
    if episode_counter%5==0:
        sensor.update(client,ros_interface,initializer,lidar,agent_i,episode_counter,create_image_message=False,create_lidar_message=False)
        time_diff=elapsed_time_entire-elapsed_time_entire_old
        distance=np.sqrt(np.power((sensor.car_state_message[5].pose.pose.position.x-porsition_old[0]), 2)+\
                         np.power((sensor.car_state_message[5].pose.pose.position.y-porsition_old[1]), 2)+\
                         np.power((sensor.car_state_message[5].pose.pose.position.z-porsition_old[2]), 2))
        speed_measurement=distance/time_diff
        speed_rate=sensor.car_state.speed/speed_measurement
        print('time_diff:',time_diff)
        print('distance:',distance)
        print('speed_measurement:',speed_measurement)
        print('speed_v:',sensor.car_state.speed)
        print('speed_rate:',speed_rate)
        print('x:',sensor.car_state_message[5].pose.pose.position.x)
        print('y:',sensor.car_state_message[5].pose.pose.position.y)
        print('z:',sensor.car_state_message[5].pose.pose.position.z)
        porsition_old[0]=sensor.car_state_message[5].pose.pose.position.x
        porsition_old[1]=sensor.car_state_message[5].pose.pose.position.y
        porsition_old[2]=sensor.car_state_message[5].pose.pose.position.z
        elapsed_time_entire_old=elapsed_time_entire