#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:50:50 2019

@author: spikezz
"""
from airsim import ImageRequest
from sensor_msgs.msg import Image,PointCloud2
import rospy
import airsim
import time
import numpy as np

rospy.init_node('publisher', anonymous=True)

pub_Image=rospy.Publisher('UnrealImage', Image, queue_size=1)
rate = rospy.Rate(100) # 10hz

client = airsim.CarClient()
client.confirmConnection()

image=True
time_step=0

while not rospy.is_shutdown():
     
    time_stamp = time.time()
    
    if image==True:
        
        responses = client.simGetImages([ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
    
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        
        try:
            
            img_rgba = img1d.reshape(response.height, response.width, 4)
#            img_rgba = np.flipud(img_rgba)
    #            airsim.write_png(os.path.normpath('greener.png'), img_rgba) 
#            img_rgba = np.flipud(img_rgba)	

            image_msg = Image()
            image_msg.height = img_rgba.shape[0];
            image_msg.width =  img_rgba.shape[1];
            image_msg.encoding = 'rgba8';
            image_msg.step = img_rgba.shape[0]*img_rgba.shape[1]*4
            image_msg.data = img_rgba.tobytes();
                
        except:
            
            print("Image acquisition failed")
            
        try:
            
            pub_Image.publish(image_msg)
            
        except:
            
            print('leer')
    
    rate.sleep()
    elapsed_time=time.time()-time_stamp
    time_step +=1
#    print('elapsed_time:',elapsed_time)