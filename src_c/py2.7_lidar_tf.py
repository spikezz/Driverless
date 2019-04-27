#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 22:55:48 2019

@author: spikezz
"""
import rospy as rp
# Because of transformations
import sys
import tf2_ros
import geometry_msgs.msg
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2,PointField
#from sensor import Lidar_Real

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)
    
def get_struct_fmt(self,is_bigendian, fields, field_names=None):
    
    fmt = '>' if is_bigendian else '<'

    offset = 0
    
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        
        if offset < field.offset:
            
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
            
        if field.datatype not in _DATATYPES:

            print(sys.stderr,'Skipping unknown PointField datatype [%d]' % (field.datatype))
            
        else:
            
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length
    
    return fmt

def handle_lidar_pose(odo_msg, lidar_frame_id):
    
    print("get odo")
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    
    t.header.stamp = rp.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = lidar_frame_id
    t.transform.translation.x = odo_msg.pose.pose.position.x
    t.transform.translation.y = odo_msg.pose.pose.position.y
    t.transform.translation.z = odo_msg.pose.pose.position.z
    t.transform.rotation.x = odo_msg.pose.pose.orientation.x
    t.transform.rotation.y = odo_msg.pose.pose.orientation.y
    t.transform.rotation.z = odo_msg.pose.pose.orientation.z
    t.transform.rotation.w = odo_msg.pose.pose.orientation.w
    
    br.sendTransform(t)
    
def handle_pointclound(lida_msg,lidar_frame_id):
    
    print("get data")
#    point_idx=int(offset/point_step)
#    print("offset end:",(offset/point_step))
#    
#    offset = 0
#    
#    for i in range(0,point_idx):
#    
#    self.point_validation.append(list(unpack_from(buff, i*point_step)))
#    
#    print("points:",points)
#    print("point_validation:",self.point_validation)
    
    
rp.init_node('lidar_position_broadcaster')
lidar_frame_id='lidar'
rate = rp.Rate(60)
#
while not rp.is_shutdown():
    print("im running")
    rp.Subscriber('Odometry_auto',Odometry,handle_lidar_pose,lidar_frame_id)
    rp.Subscriber('stitched_cloud',PointCloud2,handle_pointclound,lidar_frame_id)
#    rp.Subscriber('velodyne_points',PointCloud2,handle_lidar_pose,lidar_frame_id)
    rp.spin()
    rate.sleep()