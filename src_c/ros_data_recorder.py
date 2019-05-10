#!/usr/bin/env python

import csv
from tf.transformations import euler_from_quaternion
import rospy
import message_filters
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

def callback(state, action):
    
    with open('home/spikezz/Documents/data_log.csv', mode='a') as data_file:

        angles = euler_from_quaternion(state.pose.orientation)
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([action[0], state.pose.pose.position.x, state.pose.pose.position.y, angles[2], state.twist.twist.linear.x, state.twist.twist.linear.y, state.twist.twist.angular.z, action.data[1], action.data[3], action.data[2]])


def listener():
    
    rospy.init_node('listener', anonymous=True)
    
    state_sub = message_filters.Subscriber('Odometry_auto', Odometry)
    action_sub = message_filters.Subscriber('action', Float32MultiArray)
    
    ts = message_filters.ApproximateTimeSynchronizer([state_sub, action_sub], 10, 0.001)
    
    ts.registerCallback(callback)
    print("listening")
    rospy.spin()

listener()

