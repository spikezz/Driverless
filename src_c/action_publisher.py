#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:58:29 2019

@author: spikezz
"""
import threading_action_detecter
import rospy
#import time
from std_msgs.msg import Float32MultiArray

rospy.init_node('action_publisher', anonymous=True)
pub_action = rospy.Publisher('action', Float32MultiArray, queue_size=1000)
thread_controller=threading_action_detecter.action_detecter(1, "xbox action")

thread_controller.start()
while not rospy.is_shutdown():
#    print('printing')
    act_msg=Float32MultiArray()
    now=rospy.get_rostime()
    act_msg.data.append(now.secs)
    act_msg.data.append(now.nsecs)
    act_msg.data.append(thread_controller.throttle_signal)
    act_msg.data.append(thread_controller.brake_signal)
    act_msg.data.append(thread_controller.steering_signal)  
    pub_action.publish(act_msg)
#    print(act_msg)
#    time.sleep(1)