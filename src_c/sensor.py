#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:10:06 2019

@author: spikezz
"""
import math
import copy
#import struct
#import ctypes
import rospy
#import sys
import calculate as cal
import matplotlib.lines as mpl
import matplotlib.pyplot as plt
import numpy as np
import sensor_msgs.point_cloud2 as pcl2
from scipy import interpolate
from sensor_msgs.msg import PointCloud2,PointField

class Lidar_Real:
    
    def __init__(self,client):
        
        self.frame_point_cloud=PointCloud2()
        self.points_set=[]
        self.client = client
        self.field=PointField()

    
    def get_lidar_data(self,odm_msg):
        
        for i in range(1,2):
            
            lidar_data = self.client.getLidarData()
            
            if (len(lidar_data.point_cloud) < 3):
                
                print("\tNo points received from Lidar data")
                
            else:
    
                self.points_set = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
                self.points_set= np.reshape(self.points_set, (int(self.points_set.shape[0]/3), 3))
#                print('before:',self.points_set)
                odm = np.array([-odm_msg.pose.pose.position.x,-odm_msg.pose.pose.position.y,-odm_msg.pose.pose.position.z])
                odm_matrix=np.tile(odm, (len(self.points_set),1))
                z_correcter= np.array([-1,1,-1])
                z_correcter_matrix=np.tile(z_correcter, (len(self.points_set),1))
#                lidar_local_correcter= np.array([0,0,-0.45])
#                lidar_local_correcter_matrix=np.tile(lidar_local_correcter, (len(self.points_set),1))
#                self.points_set=(self.points_set+lidar_local_correcter_matrix)*z_correcter_matrix
                self.points_set=(self.points_set+odm_matrix)*z_correcter_matrix
#                self.points_set=self.points_set*z_correcter_matrix
#                print('after:',self.points_set)
       
                

#    
    #            print("\tReading %d: time_stamp: %d number_of_points: %d" % (i, lidar_data.time_stamp, len(self.points_set)))
    #                print("\t\tlidar position: %s" % (pprint.pformat(lidar_data.pose.position)))
    #                print("\t\tlidar orientation: %s" % (pprint.pformat(lidar_data.pose.orientation)))
                
                self.field.name="cooridnates"
                self.field.offset=0
                self.field.datatype=7
                self.field.count=3
       
                self.frame_point_cloud.header.seq=0
                self.frame_point_cloud.header.stamp=rospy.get_rostime()
                self.frame_point_cloud.header.frame_id="vld"
                
                self.frame_point_cloud = pcl2.create_cloud_xyz32(self.frame_point_cloud.header, self.points_set)
                
        return self.frame_point_cloud
    
class Sensor_Simulator(object):
    
    def __init__(self):
        
        self.lidar_bound=15
        self.cone_wide=0.1
        self.car_length=2.8
        self.cone_marker='^'
        
    def draw_point(self,x_list,y_list,ploter,cl,mk,mksize):

        point=mpl.Line2D(x_list,y_list,transform=ploter.ax.transData)
        point.set_color(cl)
        point.set_marker(mk)
        point.set_markersize(mksize)
        ploter.ax.lines.append(point)
        return point
    
    def draw_line(self,x_list,y_list,ploter,cl,mk,mksize,lw):
        
        line=mpl.Line2D(x_list,y_list,transform=ploter.ax.transData)
        line.set_color(cl)
        line.set_linewidth(lw)
        line.set_marker(mk)
        line.set_markersize(mksize)
        ploter.ax.lines.append(line)
        return line    
    
    def cover_cone(self,list_sensored_cone_sort_with_sita):
        
        list_sensored_cone_sort_with_sita_covered_free=copy.copy(list_sensored_cone_sort_with_sita)
        i=0
        
        while (i<(len(list_sensored_cone_sort_with_sita_covered_free)-1)):

            delta_sita=list_sensored_cone_sort_with_sita_covered_free[i+1][1]-list_sensored_cone_sort_with_sita_covered_free[i][1]
            half_angle_shadow_on_celestial_sphere_this=math.degrees(self.cone_wide/(list_sensored_cone_sort_with_sita_covered_free[i][0]*2))
            half_angle_shadow_on_celestial_sphere_next=math.degrees(self.cone_wide/(list_sensored_cone_sort_with_sita_covered_free[i+1][0]*2))
            
            if delta_sita<half_angle_shadow_on_celestial_sphere_this+half_angle_shadow_on_celestial_sphere_next:

                if list_sensored_cone_sort_with_sita_covered_free[i][0]>list_sensored_cone_sort_with_sita_covered_free[i+1][0]:
                    
                    list_sensored_cone_sort_with_sita_covered_free.pop(i)
                    
                else:
                    
                    list_sensored_cone_sort_with_sita_covered_free.pop(i+1)
                
                i=0   
                
            else:
                
                i+=1
        list_sensored_cone_yellow_covered_free=[]
        list_sensored_cone_blue_covered_free=[]
        
        for cone in list_sensored_cone_sort_with_sita_covered_free:
            
            if cone[3]=='y':
                
                list_sensored_cone_yellow_covered_free.append(cone)
                
            elif cone[3]=='b':
                
                list_sensored_cone_blue_covered_free.append(cone)
                
#        list_sensored_cone_yellow_covered_free=sorted(list_sensored_cone_yellow_covered_free,key=lambda x:x[1])
#        list_sensored_cone_blue_covered_free=sorted(list_sensored_cone_blue_covered_free,key=lambda x:x[1])
                
        for i in range (len(list_sensored_cone_sort_with_sita)):
            
            search_switch=False
            
            for j in range (len(list_sensored_cone_sort_with_sita_covered_free)):
                
                if list_sensored_cone_sort_with_sita_covered_free[j][2][0]==list_sensored_cone_sort_with_sita[i][2][0]\
                and list_sensored_cone_sort_with_sita_covered_free[j][2][1]==list_sensored_cone_sort_with_sita[i][2][1]:
                    
                    search_switch=True
                    break
                
            if search_switch==False:
                
                list_sensored_cone_sort_with_sita[i][3]='r'
                list_sensored_cone_sort_with_sita[i][4]='x'
                
        return list_sensored_cone_yellow_covered_free,list_sensored_cone_blue_covered_free

    def draw_cone_spline(self,list_cone,color,ploter,draw):
        
        xs=[]
        ys=[]
        theta=np.zeros(len(list_cone)+2)
        
        for c in range(len(list_cone)): 
            
            xs.append(list_cone[c].position.x_val)
            ys.append(list_cone[c].position.y_val)
            theta[c]=c
            
        xs.append(xs[0])
        xs.append(xs[1])
        ys.append(ys[0])
        ys.append(ys[1])
        theta[len(list_cone)]=len(list_cone)
        theta[len(list_cone)+1]=len(list_cone)+1
        thetanew=np.arange(theta[0],theta[-1], 0.25)

        func_x = interpolate.interp1d(theta, xs, kind='cubic')
        func_y = interpolate.interp1d(theta, ys, kind='cubic')
 
        x_new = func_x(thetanew)
        y_new = func_y(thetanew)
        index=[len(x_new)-3,len(x_new)-2,len(x_new)-1,len(x_new)]
        x_new = np.delete(x_new, index)
        y_new = np.delete(y_new, index)

        xs.pop(-1)
        ys.pop(-1)
        
        if draw:

            self.draw_line(xs,ys,ploter,color,'^',5,0)
            self.draw_line(x_new,y_new,ploter,color,None,1,1)
        
            ploter.fig = plt.gcf() 
            ploter.fig.canvas.draw() 
            ploter.fig.canvas.flush_events() 

        return [x_new,y_new]
    
    def roatat_point(self,point,eul_msg):
        
        sita_of_curve_point_and_car=cal.calculate_sita_of_radius([0,0],point[1])-math.degrees(eul_msg.z)

        if sita_of_curve_point_and_car > 180:
            
            sita_of_curve_point_and_car=sita_of_curve_point_and_car-360
            
        elif sita_of_curve_point_and_car < -180:
            
            sita_of_curve_point_and_car=sita_of_curve_point_and_car+360

        point[1]=cal.calculate_rotated_point(0,point[0],sita_of_curve_point_and_car)
    
    def rotate_sight(self,list_cone_sensored_sita,closest_yellow_curve_point_pair,closest_blue_curve_point_pair,\
                     predict_yellow_curve_point_pair,predict_blue_curve_point_pair,predict_yellow_curverature_point,\
                     predict_blue_curverature_point,eul_msg):
        
        for i in range(0,len(list_cone_sensored_sita)):
            
                list_cone_sensored_sita[i][2]=cal.calculate_rotated_point(0,\
                                       list_cone_sensored_sita[i][0],list_cone_sensored_sita[i][1])
   
        for i in range(0,len(closest_yellow_curve_point_pair)):
            
            self.roatat_point(closest_yellow_curve_point_pair[i],eul_msg)
            
        for i in range(0,len(closest_blue_curve_point_pair)):
            
            self.roatat_point(closest_blue_curve_point_pair[i],eul_msg)  
            
        for i in range(0,len(predict_yellow_curve_point_pair)):
            
            self.roatat_point(predict_yellow_curve_point_pair[i],eul_msg)
            
        for i in range(0,len(predict_blue_curve_point_pair)):
            
            self.roatat_point(predict_blue_curve_point_pair[i],eul_msg)  
            
        for i in range(0,len(predict_yellow_curverature_point)):

            self.roatat_point(predict_yellow_curverature_point[i],eul_msg) 

        for i in range(0,len(predict_blue_curverature_point)):
            
            self.roatat_point(predict_blue_curverature_point[i],eul_msg) 
    
    def update_position_cone_spline(self,odo_msg,ploter):
        
        self.draw_point([odo_msg.pose.pose.position.x],[-odo_msg.pose.pose.position.y],ploter,'g','o',5)
        
    def find_closest_curve_point_pair(self,cone_spline,odo_msg,predict_step,predict_sample_curverature,predict_step_angle):
        
        list_curve_point=[]
        list_sorted_curve_point=[]
        list_curve_point_predict=[]

        for s in range(len(cone_spline[0])):   
                
            curve_point_distance=cal.calculate_radius([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],\
                                                      [cone_spline[0][s],cone_spline[1][s]])
            
            curve_point_vector=[cone_spline[0][s]-odo_msg.pose.pose.position.x,cone_spline[1][s]-odo_msg.pose.pose.position.y]
            list_curve_point_predict.append([curve_point_distance,[curve_point_vector[0],curve_point_vector[1]],s])
            list_curve_point.append([curve_point_distance,curve_point_vector,s])
        
        list_sorted_curve_point=sorted(list_curve_point,key=lambda x:x[0])

        if list_sorted_curve_point[0][2]==(len(cone_spline[0])-1) :
            
            p_idx_0=list_sorted_curve_point[1][2]+predict_step
            p_idx_1=list_sorted_curve_point[2][2]+predict_step
        
            pa_idx_0=list_sorted_curve_point[1][2]+predict_step_angle
            pa_idx_1=list_sorted_curve_point[2][2]+predict_step_angle
            
        elif list_sorted_curve_point[1][2]==(len(cone_spline[0])-1) :
            
            p_idx_0=list_sorted_curve_point[0][2]+predict_step
            p_idx_1=list_sorted_curve_point[2][2]+predict_step
            
            pa_idx_0=list_sorted_curve_point[0][2]+predict_step_angle
            pa_idx_1=list_sorted_curve_point[2][2]+predict_step_angle
            
        else:
            
            p_idx_0=list_sorted_curve_point[0][2]+predict_step
            p_idx_1=list_sorted_curve_point[1][2]+predict_step
           
            pa_idx_0=list_sorted_curve_point[0][2]+predict_step_angle
            pa_idx_1=list_sorted_curve_point[1][2]+predict_step_angle
            
        if p_idx_0>=len(cone_spline[0]):
            
            p_idx_0=p_idx_0-len(cone_spline[0])+1
            
        if p_idx_1>=len(cone_spline[0]):    

            p_idx_1=p_idx_1-len(cone_spline[0])+1
        
        if pa_idx_0>=len(cone_spline[0]):
            
            pa_idx_0=pa_idx_0-len(cone_spline[0])+1
            
        if pa_idx_1>=len(cone_spline[0]):    

            pa_idx_1=pa_idx_1-len(cone_spline[0])+1
                
        if list_sorted_curve_point[0][2]==(len(cone_spline[0])-1) :
        
            list_curverature_predict_sorted=sorted([list_curve_point_predict[max(p_idx_0,p_idx_1)-predict_sample_curverature*2],list_curve_point_predict[max(p_idx_0,p_idx_1)-predict_sample_curverature],\
                                                                             list_curve_point_predict[max(p_idx_0,p_idx_1)]],key=lambda x:x[2])
    

            return [list_sorted_curve_point[1],list_sorted_curve_point[2]],[list_curve_point_predict[pa_idx_0],list_curve_point_predict[pa_idx_1]],\
            list_curverature_predict_sorted
        
        elif list_sorted_curve_point[1][2]==(len(cone_spline[0])-1) :
            
            list_curverature_predict_sorted=sorted([list_curve_point_predict[max(p_idx_0,p_idx_1)-predict_sample_curverature*2],list_curve_point_predict[max(p_idx_0,p_idx_1)-predict_sample_curverature],\
                                                                             list_curve_point_predict[max(p_idx_0,p_idx_1)]],key=lambda x:x[2])

            return [list_sorted_curve_point[0],list_sorted_curve_point[2]],[list_curve_point_predict[pa_idx_0],list_curve_point_predict[pa_idx_1]],\
            list_curverature_predict_sorted
        
        else:
  
            if (p_idx_0==1 and p_idx_1==len(cone_spline[0])-1) or (p_idx_1==1 and p_idx_0==len(cone_spline[0])-1) :

                list_curverature_predict_sorted=[list_curve_point_predict[len(cone_spline[0])-2],list_curve_point_predict[p_idx_1],\
                                                                                     list_curve_point_predict[p_idx_0]]

            else:
                
                list_curverature_predict_sorted=sorted([list_curve_point_predict[max(p_idx_0,p_idx_1)-predict_sample_curverature*2],list_curve_point_predict[max(p_idx_0,p_idx_1)-predict_sample_curverature],\
                                                                             list_curve_point_predict[max(p_idx_0,p_idx_1)]],key=lambda x:x[2])
      
            return [list_sorted_curve_point[0],list_sorted_curve_point[1]],[list_curve_point_predict[pa_idx_0],list_curve_point_predict[pa_idx_1]],\
            list_curverature_predict_sorted
        
    def find_sensored_cone(self,list_cone,odo_msg,eul_msg,color):

        list_sensored_cone=[]
        
        for c in range(len(list_cone)):   
 
            cone_distance=cal.calculate_radius([odo_msg.pose.pose.position.x,\
                                           odo_msg.pose.pose.position.y],[list_cone[c].position.x_val,list_cone[c].position.y_val])
            cone_vector=[list_cone[c].position.x_val-odo_msg.pose.pose.position.x,list_cone[c].position.y_val-odo_msg.pose.pose.position.y]     
            
            if cone_distance< self.lidar_bound:
                
                sita_of_cone_and_car=cal.calculate_sita_of_radius([0,0],cone_vector)-math.degrees(eul_msg.z)
                
                if sita_of_cone_and_car > 180:
                    
                    sita_of_cone_and_car=sita_of_cone_and_car-360
                    
                elif sita_of_cone_and_car < -180:
                    
                    sita_of_cone_and_car=sita_of_cone_and_car+360
                    
                if sita_of_cone_and_car<=90 and sita_of_cone_and_car>=-90:

                    list_sensored_cone.append([cone_distance,sita_of_cone_and_car,cone_vector,color,self.cone_marker])
                   
        return list_sensored_cone


        
    def plot_all(self,list_sensored_cone_sort_with_sita_covered,closest_yellow_curve_point_pair,closest_blue_curve_point_pair,\
                 predict_yellow_curve_point_pair,predict_blue_curve_point_pair,predict_yellow_curverature_point,\
                 predict_blue_curverature_point,x_rs,y_rs,ploter,eul_msg,first_person):
        
      
        
        for i in range (len(list_sensored_cone_sort_with_sita_covered)):
            
            self.draw_point([0,list_sensored_cone_sort_with_sita_covered[i][2][1]],\
                            [0,list_sensored_cone_sort_with_sita_covered[i][2][0]],\
                            ploter,list_sensored_cone_sort_with_sita_covered[i][3],list_sensored_cone_sort_with_sita_covered[i][4],5)
        
        self.draw_line([closest_yellow_curve_point_pair[0][1][1],closest_yellow_curve_point_pair[1][1][1]],[closest_yellow_curve_point_pair[0][1][0],\
                        closest_yellow_curve_point_pair[1][1][0]],ploter,'b','o',5,3)
        self.draw_line([closest_blue_curve_point_pair[0][1][1],closest_blue_curve_point_pair[1][1][1]],[closest_blue_curve_point_pair[0][1][0],\
                        closest_blue_curve_point_pair[1][1][0]],ploter,'y','o',5,3)
        
        self.draw_line([predict_yellow_curve_point_pair[0][1][1],predict_yellow_curve_point_pair[1][1][1]],[predict_yellow_curve_point_pair[0][1][0],\
                        predict_yellow_curve_point_pair[1][1][0]],ploter,'b','o',5,3)
        self.draw_line([predict_blue_curve_point_pair[0][1][1],predict_blue_curve_point_pair[1][1][1]],[predict_blue_curve_point_pair[0][1][0],\
                        predict_blue_curve_point_pair[1][1][0]],ploter,'y','o',5,3)

        self.draw_line(x_rs,y_rs,ploter,'r','o',0,1)
                
        for p in range(0,len(predict_yellow_curverature_point)):
            
            self.draw_line([predict_yellow_curverature_point[p][1][1],predict_yellow_curverature_point[p][1][1]],\
                            [predict_yellow_curverature_point[p][1][0],predict_yellow_curverature_point[p][1][0]],ploter,'y','.',5,2)
            
        for p in range(0,len(predict_blue_curverature_point)):
            
            self.draw_line([predict_blue_curverature_point[p][1][1],predict_blue_curverature_point[p][1][1]],\
                            [predict_blue_curverature_point[p][1][0],predict_blue_curverature_point[p][1][0]],ploter,'b','.',5,2)
        

        if not first_person:
            
            car_x=math.cos(eul_msg.z)*self.car_length
            car_y=math.sin(eul_msg.z)*self.car_length
            #car_point
            self.draw_line([0,car_y],[0,car_x],ploter,'r','o',5,5)
            
            lidar_right_x=math.cos(eul_msg.z+math.pi/2)*self.lidar_bound
            lidar_right_y=math.sin(eul_msg.z+math.pi/2)*self.lidar_bound
            #lidar_right_line
            self.draw_line([0,lidar_right_y],[0,lidar_right_x],ploter,'g','x',5,1)
          
            lidar_left_x=math.cos(eul_msg.z-math.pi/2)*self.lidar_bound
            lidar_left_y=math.sin(eul_msg.z-math.pi/2)*self.lidar_bound
            #lidar_left_line
            self.draw_line([0,lidar_left_y],[0,lidar_left_x],ploter,'g','x',5,1)
            
        else:

            #car_point
            self.draw_line([0,0],[0,self.car_length],ploter,'r','o',5,5)
            
            #lidar_right_line
            self.draw_line([0,self.lidar_bound],[0,0],ploter,'g','x',5,1)
            
            #lidar_left_line
            self.draw_line([0,-self.lidar_bound],[0,0],ploter,'g','x',5,1)

        ploter.fig = plt.gcf() 
        ploter.fig.canvas.draw() 
        ploter.fig.canvas.flush_events()   # update the plot and take care of window events (like resizing etc.)
        
    def find_closest_cone_pair(self,list_cone,):
    
        list_cone=sorted(list_cone,key=lambda x:x[0])

        return [list_cone[0],list_cone[1]]
    
    def calculate_distance_closest_cone_pair(self,closest_cone_pair):
        
        distance_between_cone=cal.calculate_radius(closest_cone_pair[0][2],closest_cone_pair[1][2])
        vector_closest_cone=np.array(closest_cone_pair[1][2])-np.array(closest_cone_pair[0][2])

        return distance_between_cone,vector_closest_cone
    
    def calculate_distance_curve_point_pair(self,curve_point_pair):
        
        distance_between_curve_point=cal.calculate_radius(curve_point_pair[0][1],curve_point_pair[1][1])
        vector_curve_point=np.array(curve_point_pair[1][1])-np.array(curve_point_pair[0][1])

        return distance_between_curve_point,vector_curve_point
    
    def calculate_sinus_projection_closest_curve_point_pair(self,closest_cone_pair,distance_between_cone):
        
        sin_projection=cal.calculate_projection(0,closest_cone_pair[0][0],closest_cone_pair[1][0],distance_between_cone)[1]
        
        return sin_projection
    
    def calculate_predict_angle_difference(self,predict_curve_point_pair,eul_msg):
        
        if predict_curve_point_pair[1][2]>predict_curve_point_pair[0][2]:
            
            predict_angle_difference=cal.calculate_sita_of_radius(predict_curve_point_pair[0][1],\
                                    predict_curve_point_pair[1][1])-math.degrees(eul_msg.z)
            
        else:
            
            predict_angle_difference=cal.calculate_sita_of_radius(predict_curve_point_pair[1][1],\
                                    predict_curve_point_pair[0][1])-math.degrees(eul_msg.z)

        if predict_angle_difference > 180:
            
            predict_angle_difference=predict_angle_difference-360
            
        elif predict_angle_difference < -180:
            
            predict_angle_difference=predict_angle_difference+360

        if predict_angle_difference>90:
            
            predict_angle_difference=180-predict_angle_difference
            
        elif predict_angle_difference<-90:
            
            predict_angle_difference=180+predict_angle_difference
        
        
        return predict_angle_difference