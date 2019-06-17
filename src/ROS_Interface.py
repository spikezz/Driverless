#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:47:10 2019

@author: spikezz
"""
import rospy
import airsim

import numpy as np
import Functions
from geometry_msgs.msg import Vector3, Quaternion, PoseArray,Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray,Float64
from sensor_msgs.msg import Image, PointCloud2
from airsim import ImageRequest

class ROS_Interface(object):
    
    def __init__(self,client,car_controls,remote_control):
        
        rospy.init_node('ros_msg_publisher', anonymous=True)
        self.ros_publisher=self.define_ros_publisher()
        
        if remote_control:
            print('remote_control:',remote_control)
            self.ros_subscriber=self.define_ros_subscriber(client,car_controls)
            
    def set_throttle(self,data):
        
        self.car_controls.throttle=data.data
        self.client.setCarControls(self.car_controls)
        print("throttle:",data.data)
        
    def set_brake(self,data):
        
        self.car_controls.brake=data.data
        self.client.setCarControls(self.car_controls)
        print("brake:",data.data)
        
    def set_steering(self,data):
        
        self.car_controls.steering=data.data
        self.client.setCarControls(self.car_controls)
        print("steering:",data.data)
        
    def define_ros_publisher(self):
                     
        ros_publisher={'pub_Image':rospy.Publisher('UnrealImage', Image, queue_size=15),\
                       'pub_euler':rospy.Publisher('EulerAngle', Vector3, queue_size=1),\
                       'pub_velocity':rospy.Publisher('Velocity', Vector3, queue_size=1),\
                       'pub_acceleration':rospy.Publisher('Acceleration', Vector3, queue_size=1),\
                       'pub_angular_acceleration':rospy.Publisher('AngularAcceleration', Quaternion, queue_size=1),\
                       'pub_angular_velocity':rospy.Publisher('AngularVelocity', Quaternion, queue_size=1),\
                       'pub_odometry_auto':rospy.Publisher('Odometry', Odometry, queue_size=1),\
                       'pub_action':rospy.Publisher('Action', Float32MultiArray, queue_size=1),\
                       'pub_blue_cone':rospy.Publisher('rightCones', PoseArray, queue_size=1),\
                       'pub_yellow_cone':rospy.Publisher('leftCones', PoseArray, queue_size=1),\
                       'pub_lidar_data':rospy.Publisher('LidarData', PointCloud2, queue_size=1)\
                       }
        
        return ros_publisher
    
    def define_ros_subscriber(self,client,car_controls):
        
        self.car_controls=car_controls
        self.client=client
        ros_subscriber={'sub_throttle':rospy.Subscriber("throttle",Float64,self.set_throttle),\
                       'sub_brake':rospy.Subscriber("brake",Float64,self.set_brake),\
                       'sub_steering':rospy.Subscriber("steeringAngle",Float64, self.set_steering)   
                       }
        
        return ros_subscriber
    def extracte_cone_coordinate(self,cone,cone_message):
    
        new_pose=Pose()
        new_pose.position.x=cone.position.x_val
        new_pose.position.y=cone.position.y_val
        new_pose.position.z=cone.position.z_val
        cone_message.poses.append(new_pose)

    def create_blue_cone_message(self,list_blue_cone):

        bcn_msg=PoseArray()
        
        bcn_msg.header.seq = 0
        bcn_msg.header.stamp = rospy.get_rostime()
        bcn_msg.header.frame_id = ""
      
        for cone in list_blue_cone:
            
            self.extracte_cone_coordinate(cone,bcn_msg)
         
        return bcn_msg
    
    def create_yellow_cone_message(self,list_yellow_cone):

        ycn_msg=PoseArray()
   
        ycn_msg.header.seq = 0
        ycn_msg.header.stamp = rospy.get_rostime()
        ycn_msg.header.frame_id = ""
        
        for cone in list_yellow_cone:
            
            self.extracte_cone_coordinate(cone,ycn_msg)

        return ycn_msg
    
    def create_state_message(self,client,initializer,episode_counter,create_image_message=False):
        
        car_state = client.getCarState()
        state_message=[]
        msg_time=rospy.get_rostime()
        acc_msg=Vector3()
        vel_msg=Vector3()
        a_a_msg=Quaternion()
        a_v_msg=Quaternion()
        odo_msg=Odometry()
        eul_msg=Vector3()
        euv_msg=Vector3()
        
        if create_image_message:
        
            responses = client.simGetImages([ImageRequest("0", airsim.ImageType.Scene, False, False)])
            response = responses[0]
        
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            
            try:
                
                print("episode_counter:",episode_counter%200)
                img_rgba = img1d.reshape(response.height, response.width, 3)
                
    #                if box_painter!=None:
    #                    
    #                    box_painter.img_memory[seq%200]=img_rgba
    #            airsim.write_png(os.path.normpath('greener.png'), img_rgba) 
    #            np.save('/home/spikezz/RL project copy/img_data/%d'%(seq%150), img_rgba)
                image_msg = Image()
                image_msg.header.frame_id = str(episode_counter%200)
                image_msg.height = img_rgba.shape[0];
                image_msg.width =  img_rgba.shape[1];
                image_msg.encoding = 'rgb8';
                image_msg.step = img_rgba.shape[0]*img_rgba.shape[1]*3
                image_msg.data = img_rgba.tobytes();
                    
            except:
                
                print("Image acquisition failed")    
        #0:linear_velocity,1:angular_velocity_q,2:angular_velocity_e,3:linear_acceleration,
        #4:angular_acceleration,5:odometry,6:euler_orientation,7:image
        vel_msg.x=car_state.kinematics_estimated.linear_velocity.x_val-initializer.initial_velocoty_noise[0]
        vel_msg.y=car_state.kinematics_estimated.linear_velocity.y_val-initializer.initial_velocoty_noise[1]
        vel_msg.z=car_state.kinematics_estimated.linear_velocity.z_val-initializer.initial_velocoty_noise[2]
        state_message.append(vel_msg)
    
        a_v_msg.w=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().w_val
        a_v_msg.x=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().x_val
        a_v_msg.y=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().y_val
        a_v_msg.z=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().z_val
        state_message.append(a_v_msg)
        
        euv_msg.x=car_state.kinematics_estimated.angular_velocity.x_val
        euv_msg.y=car_state.kinematics_estimated.angular_velocity.y_val
        euv_msg.z=car_state.kinematics_estimated.angular_velocity.z_val
        state_message.append(euv_msg)
        
        acc_msg.x=car_state.kinematics_estimated.linear_acceleration.x_val
        acc_msg.y=car_state.kinematics_estimated.linear_acceleration.y_val
        acc_msg.z=car_state.kinematics_estimated.linear_acceleration.z_val
        state_message.append(acc_msg)

        a_a_msg.w=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().w_val
        a_a_msg.x=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().x_val
        a_a_msg.y=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().y_val
        a_a_msg.z=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().z_val
        state_message.append(a_a_msg)
        
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
        odo_msg.twist.twist.linear.x=vel_msg.x
        odo_msg.twist.twist.linear.y=vel_msg.y
        odo_msg.twist.twist.linear.z=vel_msg.z
        odo_msg.twist.twist.angular.x=euv_msg.x
        odo_msg.twist.twist.angular.y=euv_msg.y
        odo_msg.twist.twist.angular.z=euv_msg.z
        state_message.append(odo_msg)
        
        (eul_msg.x, eul_msg.y, eul_msg.z) = Functions.euler_from_quaternion([odo_msg.pose.pose.orientation.x, \
        odo_msg.pose.pose.orientation.y, odo_msg.pose.pose.orientation.z, odo_msg.pose.pose.orientation.w])
        state_message.append(eul_msg)
        
        if create_image_message:
            
            try:
                
                state_message.append(image_msg)
                
            except:
                
                pass
            
        return state_message
            
    def create_action_message(self,car_controls):
    
        act_msg=Float32MultiArray()
        now=rospy.get_rostime()
        act_msg.data.append(now.secs)
        act_msg.data.append(now.nsecs)
        act_msg.data.append(car_controls.throttle)
        act_msg.data.append(car_controls.brake)
        act_msg.data.append(car_controls.steering)
        
        return act_msg
            
