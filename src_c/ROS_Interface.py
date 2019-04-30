#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:05:50 2019

@author: spikezz
"""
import calculate as cal
from geometry_msgs.msg import Vector3, Quaternion, PoseArray, Pose, QuaternionStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, PointCloud2

def define_ros_publisher(rp):
    
    rp.init_node('publish', anonymous=True)
                 
    ros_publisher={'pub_Image':rp.Publisher('UnrealImage', Image, queue_size=1),\
                   'pub_euler':rp.Publisher('O_to_E', Vector3, queue_size=1),\
                   'pub_velocity':rp.Publisher('velocity', Vector3, queue_size=1),\
                   'pub_acceleration':rp.Publisher('acceleration', Vector3, queue_size=1),\
                   'pub_angular_acceleration':rp.Publisher('angular_acceleration', Quaternion, queue_size=1),\
                   'pub_angular_velocity':rp.Publisher('angular_velocity', Quaternion, queue_size=1),\
                   'pub_Odometry_auto':rp.Publisher('Odometry_auto', Odometry, queue_size=1),\
                   'pub_action':rp.Publisher('action', Float32MultiArray, queue_size=1),\
                   'pub_blue_cone':rp.Publisher('rightCones', PoseArray, queue_size=1),\
                   'pub_yellow_cone':rp.Publisher('leftCones', PoseArray, queue_size=1),\
                   'pub_lidar_data':rp.Publisher('lidardata', PointCloud2, queue_size=1),\
                   'pub_Twist':rp.Publisher('optical_speed', TwistStamped, queue_size=1),\
                   'pub_Quaternion':rp.Publisher('direction_Q', QuaternionStamped, queue_size=1)
                   }
    
    return ros_publisher

def ros_car_state_message_creater(rospy,car_state,initial_velocoty_noise):
    
    ros_state_message_=[]
    msg_time=rospy.get_rostime()
    tws_msg=TwistStamped()
    qts_msg=QuaternionStamped()
    acc_msg=Vector3()
    vel_msg=Vector3()
    a_a_msg=Quaternion()
    a_v_msg=Quaternion()
    odo_msg=Odometry()
    eul_msg=Vector3()
    euv_msg=Vector3()
    
    vel_msg.x=car_state.kinematics_estimated.linear_velocity.x_val-initial_velocoty_noise[0]
    vel_msg.y=car_state.kinematics_estimated.linear_velocity.y_val-initial_velocoty_noise[1]
    vel_msg.z=car_state.kinematics_estimated.linear_velocity.z_val-initial_velocoty_noise[2]
    ros_state_message_.append(vel_msg)
    
    a_v_msg.w=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().w_val
    a_v_msg.x=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().x_val
    a_v_msg.y=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().y_val
    a_v_msg.z=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().z_val
    ros_state_message_.append(a_v_msg)
    
    (euv_msg.x,euv_msg.y,euv_msg.z)=cal.euler_from_quaternion([a_v_msg.x, a_v_msg.y, a_v_msg.z,a_v_msg.w])
    ros_state_message_.append(euv_msg)
    
    tws_msg.header.seq = 0
    tws_msg.header.stamp = msg_time
    tws_msg.header.frame_id = "optical_speed"
    tws_msg.twist.linear.x=vel_msg.x
    tws_msg.twist.linear.y=vel_msg.y
    tws_msg.twist.linear.z=vel_msg.z
    tws_msg.twist.angular.x=euv_msg.x
    tws_msg.twist.angular.y=euv_msg.y
    tws_msg.twist.angular.z=euv_msg.z
    ros_state_message_.append(tws_msg)
    
    qts_msg.header.seq = 0
    qts_msg.header.stamp = msg_time
    qts_msg.header.frame_id = "direction_Q"
    qts_msg.quaternion.w=car_state.kinematics_estimated.orientation.w_val
    qts_msg.quaternion.x=car_state.kinematics_estimated.orientation.x_val
    qts_msg.quaternion.y=car_state.kinematics_estimated.orientation.y_val
    qts_msg.quaternion.z=car_state.kinematics_estimated.orientation.z_val
    ros_state_message_.append(qts_msg)
    
    acc_msg.x=car_state.kinematics_estimated.linear_acceleration.x_val
    acc_msg.y=car_state.kinematics_estimated.linear_acceleration.y_val
    acc_msg.z=car_state.kinematics_estimated.linear_acceleration.z_val
    ros_state_message_.append(acc_msg)
    
    a_a_msg.w=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().w_val
    a_a_msg.x=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().x_val
    a_a_msg.y=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().y_val
    a_a_msg.z=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().z_val
    ros_state_message_.append(a_a_msg)
    
    odo_msg.header.seq = 0
    odo_msg.header.stamp = msg_time
    odo_msg.header.frame_id = ""
    odo_msg.pose.pose.position.x=car_state.kinematics_estimated.position.x_val
    odo_msg.pose.pose.position.y=car_state.kinematics_estimated.position.y_val
    odo_msg.pose.pose.position.z=car_state.kinematics_estimated.position.z_val
    odo_msg.pose.pose.orientation.w=car_state.kinematics_estimated.orientation.w_val
    odo_msg.pose.pose.orientation.x=car_state.kinematics_estimated.orientation.x_val
    odo_msg.pose.pose.orientation.y=car_state.kinematics_estimated.orientation.y_val
    odo_msg.pose.pose.orientation.z=car_state.kinematics_estimated.orientation.z_val
    ros_state_message_.append(odo_msg)
    
    (eul_msg.x, eul_msg.y, eul_msg.z) = cal.euler_from_quaternion([odo_msg.pose.pose.orientation.x, odo_msg.pose.pose.orientation.y, odo_msg.pose.pose.orientation.z, odo_msg.pose.pose.orientation.w])
    ros_state_message_.append(eul_msg)

    return ros_state_message_

    
def cone_coordinate_extractor(cone,cone_message):
    
    new_pose=Pose()
    new_pose.position.x=cone.position.x_val
    new_pose.position.y=cone.position.y_val
    new_pose.position.z=cone.position.z_val
    cone_message.poses.append(new_pose)

def ros_cone_message_creater(rp,list_blue_cone,list_yellow_cone):
    
    cone_message=[]
    bcn_msg=PoseArray()
    ycn_msg=PoseArray()
    
    bcn_msg.header.seq = 0
    bcn_msg.header.stamp = rp.get_rostime()
    bcn_msg.header.frame_id = ""

    ycn_msg.header.seq = 0
    ycn_msg.header.stamp = rp.get_rostime()
    ycn_msg.header.frame_id = ""
    
    for cone in list_blue_cone:
        
        cone_coordinate_extractor(cone,bcn_msg)
        
    for cone in list_yellow_cone:
        
        cone_coordinate_extractor(cone,ycn_msg)
        
    cone_message.append(bcn_msg)
    cone_message.append(ycn_msg)
    
    return cone_message
