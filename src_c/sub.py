#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:16:07 2019

@author: spikezz
"""
import rospy
from std_msgs.msg import Float32

def print_msg(data):
    print("get")
    print(data)

rospy.init_node('listener_float', anonymous=True)
sub_print=rospy.Subscriber("ssd",Float32,print_msg)

rospy.spin()