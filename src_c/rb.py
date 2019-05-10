#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:09:40 2019

@author: spikezz
"""
import rosbag
from nav_msgs.msg import Odometry
from sensor_msgs.msg import  PointCloud2
import rospy
import os.path

def write_odometry(msg):

    print("write_odometry")

    b.write('Odometry_auto',msg,t=rospy.get_rostime())
        
        
def write_lidar(msg):

    print("write_lidar")
        
    b.write('lidardata',msg,t=rospy.get_rostime())
        
    
print("start")

rospy.init_node('subscriber', anonymous=True)

if os.path.exists('/home/spikezz/RL project copy/Driverless/src_c/airsim_lidar_data1.bag'):
    
    b = rosbag.Bag('airsim_lidar_data1.bag', 'a')
    
else:
    
    b = rosbag.Bag('airsim_lidar_data1.bag', 'w')

#try:
#    
#    b = rosbag.Bag('airsim_lidar_data1.bag', 'a')
#except:
#    print("fehler")
sub_Odometry=rospy.Subscriber("Odometry_auto",Odometry,write_odometry)
sub_lidar=rospy.Subscriber("lidardata",PointCloud2,write_lidar)
rospy.spin()
