#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:43:52 2019

@author: spikezz
"""
import Functions
import rospy
import math
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mpl
import sensor_msgs.point_cloud2 as pcl2

from sensor_msgs.msg import PointCloud2,PointField

class Sensor_Box(object):
        
    class Sensor_Visualizer(object):
        
        class Top_View_Ploter(object):
        
            def __init__(self,xlim,ylim,fig_size):
                
                self.fig = plt.figure(figsize=fig_size)
#                self.fig.show()
                self.ax = self.fig.add_subplot(1,1,1)
                self.fig.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9)
                self.ax.set_xlim(xlim[0], xlim[1])
                self.ax.set_ylim(ylim[0], ylim[1]) 
                plt.ion()
            
            def draw_line(self,x_list,y_list,ploter,cl,mk,mksize,lw,lb):
        
                line=mpl.Line2D(x_list,y_list,alpha=0.5,transform=ploter.ax.transData)
                line.set_color(cl)
                line.set_linewidth(lw)
                line.set_marker(mk)
                line.set_markersize(mksize)
                if lb!=None:
                    line.set_label(lb)
                ploter.ax.lines.append(line)
                
                return line    
                    
            def clear(self):
                
                self.ax.lines.clear()
                
            def roatat_point(self,point,eul_msg):
        
                sita_of_curve_point_and_car=Functions.calculate_sita_of_radius([0,0],point[1])-math.degrees(eul_msg.z)
        
                if sita_of_curve_point_and_car > 180:
                    
                    sita_of_curve_point_and_car=sita_of_curve_point_and_car-360
                    
                elif sita_of_curve_point_and_car < -180:
                    
                    sita_of_curve_point_and_car=sita_of_curve_point_and_car+360
        
                point[1]=Functions.calculate_rotated_point(0,point[0],sita_of_curve_point_and_car)
                
            def rotate_sight(self,list_cone_sensored_sita,closest_yellow_curve_point_pair,closest_blue_curve_point_pair,\
                     predict_yellow_curve_point_pair,predict_blue_curve_point_pair,predict_yellow_curverature_point,\
                     predict_blue_curverature_point,velocity_2d_correction,eul_msg):

                self.roatat_point(velocity_2d_correction,eul_msg)
                
                for i in range(0,len(list_cone_sensored_sita)):
                    
                        list_cone_sensored_sita[i][2]=Functions.calculate_rotated_point(0,list_cone_sensored_sita[i][0],list_cone_sensored_sita[i][1])
           
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
    
        def __init__(self,draw_sight):
            
            self.sensor_boundary=15
            
            self.cone_marker='^'
            self.cone_wide=0.1
            
            self.car_length=2.8
            self.track_width=5
            
            if draw_sight:
                
                self.lidar_top_view=self.Top_View_Ploter([-15,15],[-15,15],[6.4, 4.8])
        
        def update(self,cone_set,sensor,agent_controller,agent_imitation,curverature_meter,draw_sight=False,first_person=True):
            
            self.list_yellow_cone_sensored_sita=self.find_sensored_cone(cone_set.list_yellow_cone,sensor.car_state_message[5],sensor.car_state_message[6],color='y')
            self.list_blue_cone_sensored_sita= self.find_sensored_cone(cone_set.list_blue_cone,sensor.car_state_message[5],sensor.car_state_message[6],color='b')
            
            self.list_cone_sensored_sita=copy.deepcopy(self.list_yellow_cone_sensored_sita)
            self.list_cone_sensored_sita=self.list_cone_sensored_sita+self.list_blue_cone_sensored_sita
            self.list_cone_sensored_sita=sorted(self.list_cone_sensored_sita,key=lambda x:x[1])
            
            self.list_sensored_cone_yellow_covered_free,self.list_sensored_cone_blue_covered_free=self.cover_cone(self.list_cone_sensored_sita)
            self.list_sensored_cone_covered_free=self.list_sensored_cone_yellow_covered_free+self.list_sensored_cone_blue_covered_free
            
            if len(self.list_sensored_cone_yellow_covered_free)>=2:
            
                self.closest_yellow_cone_pair=self.find_closest_cone_pair(self.list_sensored_cone_yellow_covered_free)
                self.distance_between_yellow_cone,self.vector_closest_yellow_cone=self.calculate_distance_closest_cone_pair(self.closest_yellow_cone_pair)
                
            else:
    
                if len(self.list_sensored_cone_yellow_covered_free)==1:
    
                    self.closest_yellow_cone=self.list_sensored_cone_yellow_covered_free
                    print("single yellow cone")
                    
                else:
                    
                    print("no yellow cone detected")
                    
            if len(self.list_sensored_cone_blue_covered_free)>=2:
                
                self.closest_blue_cone_pair=self.find_closest_cone_pair(self.list_sensored_cone_blue_covered_free)
                self.distance_between_blue_cone,self.vector_closest_blue_yellow_cone=self.calculate_distance_closest_cone_pair(self.closest_blue_cone_pair)
                
            else:
    
                if len(self.list_sensored_cone_blue_covered_free)==1:
                    
                    self.closest_blue_cone=self.list_sensored_cone_blue_covered_free
                    print("single blue cone")
                    
                else:
                    
                    print("no blue cone detected")
                
            self.closest_yellow_curve_point_pair,self.predict_yellow_curve_point_pair,self.predict_yellow_curverature_point=\
            self.find_closest_curve_point_pair(cone_set.yellow_cone_spline,sensor.car_state_message[5],agent_controller.predict_step,\
                                               agent_controller.curverature_sample_step,agent_controller.predict_step_angle)
        
            self.distance_between_closet_yellow_curve_point,self.vector_closest_yellow_curve_point=\
            self.calculate_distance_curve_point_pair(self.closest_yellow_curve_point_pair)    
            
            self.distance_between_predict_yellow_curve_point,self.vector_predict_yellow_curve_point=\
            self.calculate_distance_curve_point_pair(self.predict_yellow_curve_point_pair)
            
            self.closest_blue_curve_point_pair,self.predict_blue_curve_point_pair,self.predict_blue_curverature_point=\
            self.find_closest_curve_point_pair(cone_set.blue_cone_spline,sensor.car_state_message[5],agent_controller.predict_step,\
                                               agent_controller.curverature_sample_step,agent_controller.predict_step_angle)
            
            self.distance_between_closet_blue_curve_point,self.vector_closest_blue_curve_point=\
            self.calculate_distance_curve_point_pair(self.closest_blue_curve_point_pair)
            
            self.distance_between_predict_blue_curve_point,self.vector_predict_blue_curve_point=\
            self.calculate_distance_curve_point_pair(self.predict_blue_curve_point_pair)    
        
            self.predict_angle_difference=self.calculate_predict_angle_difference(self.predict_yellow_curve_point_pair,sensor.car_state_message[6])
            
            if self.distance_between_predict_yellow_curve_point<self.distance_between_closet_blue_curve_point:
            
                self.curverature_sign=-1
                
            elif self.distance_between_predict_yellow_curve_point>=self.distance_between_closet_blue_curve_point:
                
                self.curverature_sign=1
            
            try:
            
                self.sin_projection_yellow=self.calculate_sinus_projection_closest_curve_point_pair(self.closest_yellow_curve_point_pair,\
                                                                                                  self.distance_between_closet_yellow_curve_point)
                self.sin_projection_blue=self.calculate_sinus_projection_closest_curve_point_pair(self.closest_blue_curve_point_pair,\
                                                                                                self.distance_between_closet_blue_curve_point)
                self.sin_projection_difference=self.sin_projection_yellow-self.sin_projection_blue
  
                if self.sin_projection_yellow>self.track_width:
                    
                    self.sin_projection_blue=self.sin_projection_blue*(-1)
                    
                if self.sin_projection_blue>self.track_width:
                    
                    self.sin_projection_yellow=self.sin_projection_yellow*(-1)
                    
                self.mittle_position=self.sin_projection_yellow-self.track_width/2
#                print("sin_projection_yellow:%.2f \t sin_projection_blue:%.2f"%(self.sin_projection_yellow,self.sin_projection_blue))
#                print("sin_projection difference:%.2f"%(self.sin_projection_difference))
               
            except:
                
                print("shit happens")
                
            agent_imitation.lateral_position.append(self.mittle_position)
            agent_imitation.set_point_lateral_position.append(agent_controller.set_point_steering)
            
            self.predict_yellow_curverature_point_c=copy.deepcopy(self.predict_yellow_curverature_point)
            self.predict_blue_curverature_point_c=copy.deepcopy(self.predict_blue_curverature_point)
            
            if draw_sight: 
                
                if first_person:
                    
                    self.lidar_top_view.rotate_sight(self.list_cone_sensored_sita,self.closest_yellow_curve_point_pair,self.closest_blue_curve_point_pair,\
                                         self.predict_yellow_curve_point_pair,self.predict_blue_curve_point_pair,self.predict_yellow_curverature_point_c,\
                                         self.predict_blue_curverature_point_c,sensor.velocity_2d_correction,sensor.car_state_message[6])  
            
            x_center_yellow,y_center_yellow,r_yellow,circle_norm_vector_yellow=Functions.calculate_circle_curverature_with_3p([self.predict_yellow_curverature_point_c[0][1][1],\
                                                                    self.predict_yellow_curverature_point_c[1][1][1],self.predict_yellow_curverature_point_c[2][1][1]],\
                                                                    [self.predict_yellow_curverature_point_c[0][1][0],self.predict_yellow_curverature_point_c[1][1][0],\
                                                                     self.predict_yellow_curverature_point_c[2][1][0]])
        
            x_center_blue,y_center_blue,r_blue,circle_norm_vector_blue=Functions.calculate_circle_curverature_with_3p([self.predict_blue_curverature_point_c[0][1][1],\
                                                                    self.predict_blue_curverature_point_c[1][1][1],self.predict_blue_curverature_point_c[2][1][1]],\
                                                                    [self.predict_blue_curverature_point_c[0][1][0],self.predict_blue_curverature_point_c[1][1][0],\
                                                                     self.predict_blue_curverature_point_c[2][1][0]])
            
            agent_imitation.predict_curverature_measured.append((1/r_yellow+1/r_blue)/2)
            curverature_meter.measure_curverature(agent_controller,agent_imitation)
            
            if draw_sight:
                    
                x_rs,y_rs = Functions.calculate_circle(x_center_yellow,y_center_yellow,r_yellow)
                self.plot_all(self.list_cone_sensored_sita,self.closest_yellow_curve_point_pair,self.closest_blue_curve_point_pair,\
                                 self.predict_yellow_curve_point_pair,self.predict_blue_curve_point_pair,self.predict_yellow_curverature_point_c,\
                                 self.predict_blue_curverature_point_c,x_rs,y_rs,self.lidar_top_view,sensor,sensor.car_state_message[6],first_person)
            
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
        
        def find_sensored_cone(self,list_cone,odo_msg,eul_msg,color):

            list_sensored_cone=[]
            
            for c in range(len(list_cone)):   
     
                cone_distance=Functions.calculate_radius([odo_msg.pose.pose.position.x,\
                                               odo_msg.pose.pose.position.y],[list_cone[c].position.x_val,list_cone[c].position.y_val])
                cone_vector=[list_cone[c].position.x_val-odo_msg.pose.pose.position.x,list_cone[c].position.y_val-odo_msg.pose.pose.position.y]     
                
                if cone_distance< self.sensor_boundary:
                    
                    sita_of_cone_and_car=Functions.calculate_sita_of_radius([0,0],cone_vector)-math.degrees(eul_msg.z)
                    
                    if sita_of_cone_and_car > 180:
                        
                        sita_of_cone_and_car=sita_of_cone_and_car-360
                        
                    elif sita_of_cone_and_car < -180:
                        
                        sita_of_cone_and_car=sita_of_cone_and_car+360
                        
                    if sita_of_cone_and_car<=90 and sita_of_cone_and_car>=-90:
    
                        list_sensored_cone.append([cone_distance,sita_of_cone_and_car,cone_vector,color,self.cone_marker])
                       
            return list_sensored_cone
        
        def find_closest_curve_point_pair(self,cone_spline,odo_msg,predict_step,curverature_sample_step,predict_step_angle):
            
            list_curve_point=[]
            list_sorted_curve_point=[]
            list_curve_point_predict=[]
    
            for s in range(len(cone_spline[0])):   
                    
                curve_point_distance=Functions.calculate_radius([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[cone_spline[0][s],cone_spline[1][s]])
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
            
                list_curverature_predict_sorted=sorted([list_curve_point_predict[max(p_idx_0,p_idx_1)-curverature_sample_step*2],\
                list_curve_point_predict[max(p_idx_0,p_idx_1)-curverature_sample_step],list_curve_point_predict[max(p_idx_0,p_idx_1)]],\
                key=lambda x:x[2])
        
  
                return [list_sorted_curve_point[1],list_sorted_curve_point[2]],[list_curve_point_predict[pa_idx_0],list_curve_point_predict[pa_idx_1]],\
                list_curverature_predict_sorted
            
            elif list_sorted_curve_point[1][2]==(len(cone_spline[0])-1) :
                
                list_curverature_predict_sorted=sorted([list_curve_point_predict[max(p_idx_0,p_idx_1)-curverature_sample_step*2],\
                                                list_curve_point_predict[max(p_idx_0,p_idx_1)-curverature_sample_step],\
                                                list_curve_point_predict[max(p_idx_0,p_idx_1)]],key=lambda x:x[2])

                return [list_sorted_curve_point[0],list_sorted_curve_point[2]],[list_curve_point_predict[pa_idx_0],list_curve_point_predict[pa_idx_1]],\
                list_curverature_predict_sorted
            
            else:
      
                if (p_idx_0==1 and p_idx_1==len(cone_spline[0])-1) or (p_idx_1==1 and p_idx_0==len(cone_spline[0])-1) :
    
                    list_curverature_predict_sorted=[list_curve_point_predict[len(cone_spline[0])-2],list_curve_point_predict[p_idx_1],\
                                                                                         list_curve_point_predict[p_idx_0]]
    
                else:
                    
                    list_curverature_predict_sorted=sorted([list_curve_point_predict[max(p_idx_0,p_idx_1)-curverature_sample_step*2],\
                                                    list_curve_point_predict[max(p_idx_0,p_idx_1)-curverature_sample_step],\
                                                    list_curve_point_predict[max(p_idx_0,p_idx_1)]],key=lambda x:x[2])
     
                return [list_sorted_curve_point[0],list_sorted_curve_point[1]],[list_curve_point_predict[pa_idx_0],list_curve_point_predict[pa_idx_1]],\
                list_curverature_predict_sorted
                
        def find_closest_cone_pair(self,list_cone,):
        
            list_cone=sorted(list_cone,key=lambda x:x[0])
            
            return [list_cone[0],list_cone[1]]
        
        def plot_all(self,list_sensored_cone_sort_with_sita_covered,closest_yellow_curve_point_pair,closest_blue_curve_point_pair,\
                 predict_yellow_curve_point_pair,predict_blue_curve_point_pair,predict_yellow_curverature_point,\
                 predict_blue_curverature_point,x_rs,y_rs,ploter,sensor,eul_msg,first_person):
        
      
        
            for i in range (len(list_sensored_cone_sort_with_sita_covered)):
                
                self.lidar_top_view.draw_line([0,list_sensored_cone_sort_with_sita_covered[i][2][1]],\
                                [0,list_sensored_cone_sort_with_sita_covered[i][2][0]],\
                                ploter,list_sensored_cone_sort_with_sita_covered[i][3],list_sensored_cone_sort_with_sita_covered[i][4],5,0.5,None)
            #
            self.lidar_top_view.draw_line([self.closest_yellow_cone_pair[0][2][1],self.closest_yellow_cone_pair[1][2][1]],\
                                          [self.closest_yellow_cone_pair[0][2][0],self.closest_yellow_cone_pair[1][2][0]],ploter,'b','$yellow\ cone\ vector$',50,0.5,None)
            
            self.lidar_top_view.draw_line([self.closest_blue_cone_pair[0][2][1],self.closest_blue_cone_pair[1][2][1]],\
                                          [self.closest_blue_cone_pair[0][2][0],self.closest_blue_cone_pair[1][2][0]],ploter,'y','$blue\ cone\ vector$',50,0.5,None)
            #projection point
            self.lidar_top_view.draw_line([closest_yellow_curve_point_pair[0][1][1],closest_yellow_curve_point_pair[1][1][1]],\
                                          [closest_yellow_curve_point_pair[0][1][0], closest_yellow_curve_point_pair[1][1][0]],ploter,'b','o',5,3,None)
            self.lidar_top_view.draw_line([closest_blue_curve_point_pair[0][1][1],closest_blue_curve_point_pair[1][1][1]],\
                                          [closest_blue_curve_point_pair[0][1][0],closest_blue_curve_point_pair[1][1][0]],ploter,'y','o',5,3,None)
                        
            #predict point
            self.lidar_top_view.draw_line([predict_yellow_curve_point_pair[0][1][1],predict_yellow_curve_point_pair[1][1][1]],\
                                          [predict_yellow_curve_point_pair[0][1][0],predict_yellow_curve_point_pair[1][1][0]],ploter,'b','o',5,3,None)
            self.lidar_top_view.draw_line([predict_blue_curve_point_pair[0][1][1],predict_blue_curve_point_pair[1][1][1]],\
                                          [predict_blue_curve_point_pair[0][1][0],predict_blue_curve_point_pair[1][1][0]],ploter,'y','o',5,3,None)
                        
            #curverature circle
            self.lidar_top_view.draw_line(x_rs,y_rs,ploter,'r','o',0,1,None)
                    
            for p in range(0,len(predict_yellow_curverature_point)):
                
                self.lidar_top_view.draw_line([predict_yellow_curverature_point[p][1][1],predict_yellow_curverature_point[p][1][1]],\
                                [predict_yellow_curverature_point[p][1][0],predict_yellow_curverature_point[p][1][0]],ploter,'y','.',5,2,None)
                
            for p in range(0,len(predict_blue_curverature_point)):
                
                self.lidar_top_view.draw_line([predict_blue_curverature_point[p][1][1],predict_blue_curverature_point[p][1][1]],\
                                [predict_blue_curverature_point[p][1][0],predict_blue_curverature_point[p][1][0]],ploter,'b','.',5,2,None)
            
    
            if not first_person:
                
         
                #velocity
                self.lidar_top_view.draw_line([0,sensor.car_state_message[0].y],[0,sensor.car_state_message[0].x],ploter,'aqua','d',5,0.5,None)
                

                car_x=math.cos(eul_msg.z)*self.car_length*0.5
                car_y=math.sin(eul_msg.z)*self.car_length*0.5
                #car_point
                self.lidar_top_view.draw_line([-car_y,car_y],[-car_x,car_x],ploter,'r','o',5,12,None)

                car_x=math.cos(eul_msg.z)*self.car_length*0.5
                car_y=math.sin(eul_msg.z)*self.car_length*0.5
                #car_point
                self.lidar_top_view.draw_line([-car_y,car_y],[-car_x,car_x],ploter,'r','o',5,12,None)
                
                lidar_front_x=math.cos(eul_msg.z)*self.sensor_boundary
                lidar_front_y=math.sin(eul_msg.z)*self.sensor_boundary
                #lidar_front_line
                self.lidar_top_view.draw_line([0,lidar_front_y],[0,lidar_front_x],ploter,'g','x',5,1,None)
                
                lidar_right_x=math.cos(eul_msg.z+math.pi/2)*self.sensor_boundary
                lidar_right_y=math.sin(eul_msg.z+math.pi/2)*self.sensor_boundary
                #lidar_right_line
                self.lidar_top_view.draw_line([0,lidar_right_y],[0,lidar_right_x],ploter,'g','x',5,1,None)
              
                lidar_left_x=math.cos(eul_msg.z-math.pi/2)*self.sensor_boundary
                lidar_left_y=math.sin(eul_msg.z-math.pi/2)*self.sensor_boundary
                #lidar_left_line
                self.lidar_top_view.draw_line([0,lidar_left_y],[0,lidar_left_x],ploter,'g','x',5,1,None)
                
            else:
                
                #velocity
                self.lidar_top_view.draw_line([0,sensor.velocity_2d_correction[1][1]],[0,sensor.velocity_2d_correction[1][0]],ploter,'aqua','d',5,0.5,None)
                
                #car_point
                self.lidar_top_view.draw_line([0,0],[-self.car_length*0.5,self.car_length*0.5],ploter,'r','o',5,12,None)
                
                #lidar_front_line
                self.lidar_top_view.draw_line([0,0],[0,self.sensor_boundary],ploter,'g','x',5,1,None)
                
                #lidar_right_line
                self.lidar_top_view.draw_line([0,self.sensor_boundary],[0,0],ploter,'g','x',5,1,None)
                
                #lidar_left_line
                self.lidar_top_view.draw_line([0,-self.sensor_boundary],[0,0],ploter,'g','x',5,1,None)
    
            ploter.fig = plt.gcf() 
            ploter.fig.canvas.draw() 
            ploter.fig.canvas.flush_events()   # update the plot and take care of window events (like resizing etc.)
            
        def calculate_distance_closest_cone_pair(self,closest_cone_pair):
            
            distance_between_cone=Functions.calculate_radius(closest_cone_pair[0][2],closest_cone_pair[1][2])
            vector_closest_cone=np.array(closest_cone_pair[1][2])-np.array(closest_cone_pair[0][2])
    
            return distance_between_cone,vector_closest_cone
        
        def calculate_distance_curve_point_pair(self,curve_point_pair):
        
            distance_between_curve_point=Functions.calculate_radius(curve_point_pair[0][1],curve_point_pair[1][1])
            vector_curve_point=np.array(curve_point_pair[1][1])-np.array(curve_point_pair[0][1])
    
            return distance_between_curve_point,vector_curve_point
        
        def calculate_predict_angle_difference(self,predict_curve_point_pair,eul_msg):
        
            if predict_curve_point_pair[1][2]>predict_curve_point_pair[0][2]:
                
                predict_angle_difference=Functions.calculate_sita_of_radius(predict_curve_point_pair[0][1],\
                                        predict_curve_point_pair[1][1])-math.degrees(eul_msg.z)
                
            else:
                
                predict_angle_difference=Functions.calculate_sita_of_radius(predict_curve_point_pair[1][1],\
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
        
        def calculate_sinus_projection_closest_curve_point_pair(self,closest_cone_pair,distance_between_cone):
        
            sin_projection=Functions.calculate_projection(0,closest_cone_pair[0][0],closest_cone_pair[1][0],distance_between_cone)[1]
            
            return sin_projection
        
    class Real_Lidar(object):
        
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
    #                odm = np.array([-odm_msg.pose.pose.position.x,-odm_msg.pose.pose.position.y,-odm_msg.pose.pose.position.z])
    #                odm_matrix=np.tile(odm, (len(self.points_set),1))
                    z_correcter= np.array([-1,1,-1])
                    z_correcter_matrix=np.tile(z_correcter, (len(self.points_set),1))
    #                lidar_local_correcter= np.array([0,0,-0.45])
    #                lidar_local_correcter_matrix=np.tile(lidar_local_correcter, (len(self.points_set),1))
    #                self.points_set=(self.points_set+lidar_local_correcter_matrix)*z_correcter_matrix
    #                self.points_set=(self.points_set+odm_matrix)*z_correcter_matrix
                    self.points_set=self.points_set*z_correcter_matrix
                    self.field.name="cooridnates"
                    self.field.offset=0
                    self.field.datatype=7
                    self.field.count=3
                    self.frame_point_cloud.header.seq=0
                    self.frame_point_cloud.header.stamp=rospy.get_rostime()
                    self.frame_point_cloud.header.frame_id="karat_lidar"
                    self.frame_point_cloud = pcl2.create_cloud_xyz32(self.frame_point_cloud.header, self.points_set)
                    
            return self.frame_point_cloud
        
        def update(self):
            pass
            
    class Curverature_Meter(object):
        
        def __init__(self):
            
            self.curv_min=1
            self.curv_max=0
            self.curverature_flatten_grad=16
            
        def measure_curverature(self,agent_controller,agent_imitation):
            
            if len(agent_imitation.predict_curverature_measured)>(agent_controller.curverature_sample_step+self.curverature_flatten_grad):
                
                sum_flattened_predict_curverature=0
                sum_flattened_predict_curverature_gradient=0
                
                for i in range(0,agent_controller.curverature_sample_step+self.curverature_flatten_grad):
           
                    sum_flattened_predict_curverature=agent_imitation.predict_curverature_measured[-(i+1)]+sum_flattened_predict_curverature
                    
                    if i<(agent_controller.curverature_sample_step+self.curverature_flatten_grad-1):
                        
                        sum_flattened_predict_curverature_gradient=np.abs(agent_imitation.predict_curverature_measured[-(i+1)]-\
                        agent_imitation.predict_curverature_measured[-(i+2)])+sum_flattened_predict_curverature_gradient
             
                self.flattened_predict_curverature=sum_flattened_predict_curverature/(agent_controller.curverature_sample_step+self.curverature_flatten_grad*(1+sum_flattened_predict_curverature_gradient)**1)
                
                if self.flattened_predict_curverature<self.curv_min:
                    
                    self.curv_min=self.flattened_predict_curverature
                    
                if self.flattened_predict_curverature>self.curv_max:
                    
                    self.curv_max=self.flattened_predict_curverature
                
                agent_imitation.predict_curverature.append(self.flattened_predict_curverature)
            
    def __init__(self,client,ros_interface,initializer,lidar,episode_counter):
        
        self.car_state = client.getCarState()
        self.car_state_message=ros_interface.create_state_message(client,initializer,episode_counter,create_image_message=False)
        
    def update(self,client,ros_interface,initializer,lidar,agent_imitation,episode_counter,create_image_message,create_lidar_message):
        
        self.car_state = client.getCarState()
        self.car_state_message=ros_interface.create_state_message(client,initializer,episode_counter,create_image_message=create_image_message)
        
        speed_optic=np.sqrt(np.power(self.car_state_message[0].x, 2)+np.power(self.car_state_message[0].y, 2)+np.power(self.car_state_message[0].z, 2))
        
        if speed_optic>30:
            
            speed_optic=0
            
        agent_imitation.optic_speed.append(speed_optic)
        self.velocity_2d_correction=[speed_optic,[self.car_state_message[0].x,self.car_state_message[0].y]]
        if create_lidar_message:
            
            self.pointcloud_message=lidar.get_lidar_data(self.car_state_message[5])

        