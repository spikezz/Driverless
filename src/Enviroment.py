#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:08:37 2019

@author: spikezz
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mpl
#import ROS_Interface as RI

from scipy import interpolate

class Cone_set(object):
 
    class Global_Map_Ploter(object):
    
        def __init__(self,xlim,ylim,fig_size):
            
            self.fig = plt.figure(figsize=fig_size)
#            self.fig.show()
            self.ax = self.fig.add_subplot(1,1,1)
            self.fig.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9)
            self.ax.set_xlim(xlim[0], xlim[1])
            self.ax.set_ylim(ylim[0], ylim[1])
            plt.ion()
            
        def draw_line(self,x_list,y_list,ploter,cl,mk,mksize,lw):
            
            line=mpl.Line2D(x_list,y_list,transform=ploter.ax.transData)
            line.set_color(cl)
            line.set_linewidth(lw)
            line.set_marker(mk)
            line.set_markersize(mksize)
            ploter.ax.lines.append(line)
            
            return line         
    
        def update_auto_position(self,odo_msg,ploter):
            
            self.draw_line([odo_msg.pose.pose.position.x],[-odo_msg.pose.pose.position.y],ploter,'g','o',5,0)
            ploter.fig = plt.gcf() 
            ploter.fig.canvas.draw() 
            ploter.fig.canvas.flush_events() 
            
        def update(self,odo_msg):
            
            if len(self.ax.lines)>4:

                self.ax.lines.pop(-1)
            
            self.update_auto_position(odo_msg,self)
            
    def __init__(self,client,initializer,ros_interface,sensor_visualizer,agent_controller,draw_map=True,loop_circuit=True):
           
        #cone set of different color  
        list_yellow_cone_random=client.simGetObjectPosesWithTag("yCone")  
        
        while list_yellow_cone_random==[]:

            list_yellow_cone_random=client.simGetObjectPosesWithTag("yCone")  
            
        list_blue_cone_random=client.simGetObjectPosesWithTag("bCone") 
        
        while list_blue_cone_random==[]:

            list_blue_cone_random=client.simGetObjectPosesWithTag("bCone")
            
        #end cone
        if not loop_circuit:
            
            self.coneback=client.simGetObjectPosesWithTag("finish")
            
            while self.coneback==[]:

                self.coneback=client.simGetObjectPosesWithTag("finish")
        #calculate
        self.list_blue_cone=[]
        self.list_yellow_cone=[]
        #display
        self.list_blue_cone_display=[]
        self.list_yellow_cone_display=[]
        #take cone with index
        for i in range(0,len(list_yellow_cone_random)):
        
            blue_cone=client.simGetObjectPosesWithTag('blue_'+str(i))
            
            while blue_cone==[]:
                
                blue_cone=client.simGetObjectPosesWithTag('blue_'+str(i))
                
            yellow_cone=client.simGetObjectPosesWithTag('yellow_'+str(i))
            
            while yellow_cone==[]:
                
                yellow_cone=client.simGetObjectPosesWithTag('yellow_'+str(i))
                
            
            self.list_blue_cone.append(blue_cone[0])
            self.list_yellow_cone.append(yellow_cone[0])
            
            self.blue_cone_correction=copy.deepcopy(blue_cone[0])
            self.yellow_cone_correction=copy.deepcopy(yellow_cone[0])
            
            #correction of the Unreal raw corrodinate for display on virsualizer
            self.blue_cone_correction.position.y_val=(-1)*blue_cone[0].position.y_val
            self.yellow_cone_correction.position.y_val=(-1)*yellow_cone[0].position.y_val
            self.list_blue_cone_display.append(self.blue_cone_correction)
            self.list_yellow_cone_display.append(self.yellow_cone_correction)
          
        self.blue_cone_message=ros_interface.create_blue_cone_message(self.list_blue_cone)  
        self.yellow_cone_message=ros_interface.create_yellow_cone_message(self.list_yellow_cone)  
        
        if draw_map:
            
            self.global_map=self.Global_Map_Ploter([-300,-50],[-100,150],[12.8, 9.6])
            
            #display
            self.blue_cone_spline_display=self.draw_cone_spline(self.list_blue_cone_display,'b',self.global_map,draw=draw_map)
            self.yellow_cone_spline_display=self.draw_cone_spline(self.list_yellow_cone_display,'y',self.global_map,draw=draw_map)
            
        #calculate
        self.blue_cone_spline=self.draw_cone_spline(self.list_blue_cone,'b',ploter=None,draw=False)
        self.yellow_cone_spline=self.draw_cone_spline(self.list_yellow_cone,'y',ploter=None,draw=False)
        
        self.start_point=self.record_start_point(client,initializer,ros_interface,sensor_visualizer,agent_controller,episode_counter=0)
        
        print('start_point:',self.start_point)
        print("length_idx_blue",len(self.blue_cone_spline[0]))
        print("length_idx_yellow",len(self.yellow_cone_spline[0]))
        
    def record_start_point(self,client,initializer,ros_interface,sensor_visualizer,agent_controller,episode_counter):
        
        car_state_message=ros_interface.create_state_message(client,initializer,episode_counter,create_image_message=False)
        closest_yellow_curve_point_pair,predict_yellow_curve_point_pair,predict_yellow_curverature_point= \
        sensor_visualizer.find_closest_curve_point_pair(self.yellow_cone_spline,car_state_message[5],episode_counter,\
        agent_controller.curverature_sample_step,agent_controller.predict_step_angle)
        start_point=closest_yellow_curve_point_pair[0][2]
        
        return start_point
    
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
            
            print('drawing spline......')
            self.global_map.draw_line(xs,ys,ploter,color,'^',5,0)
            self.global_map.draw_line(x_new,y_new,ploter,color,None,1,1)
        
            ploter.fig = plt.gcf() 
            ploter.fig.canvas.draw() 
            ploter.fig.canvas.flush_events() 

        return [x_new,y_new]
        