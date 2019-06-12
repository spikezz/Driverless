#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:17:00 2019

@author: spikezz
"""
import matplotlib.pyplot as plt

class Summary_Scope(object):
    
    class Summary_Ploter(object):
    
        def __init__(self, row, column):
            
            self.fig = plt.figure()
            self.fig.show()
            self.ax=[None]*(row*column)
            
            for i in range(0,row*column):
                
                self.ax[i] = self.fig.add_subplot(row,column,i+1)
                
            self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95) 
            plt.ion()
            
    def __init__(self,plot_action=False,plot_speed=False,plot_time=False):
        
        if plot_action:
            
            self.action_ploter=self.Summary_Ploter(row=1, column=3)
        
        if plot_speed:
            
            self.speed_ploter=self.Summary_Ploter(row=1, column=1)
        
        if plot_time:
            
            self.time_ploter=self.Summary_Ploter(row=1, column=1)
            
    def plot_summary(self,agent):

        #accelerate
        if hasattr(self, 'action_ploter'):
            
            self.action_ploter.ax[0].clear()
            self.action_ploter.ax[0].plot(agent.throttle_controller,'r',linewidth=0.5,label='throttle controller')
            self.action_ploter.ax[0].plot(agent.throttle_imitation,'g',linewidth=0.5,label='throttle imitation')
            self.action_ploter.ax[0].plot(agent.diff_throttle,'b',linewidth=0.5,label='throttle difference')
            self.action_ploter.ax[0].legend(loc='lower left')
            self.action_ploter.ax[0].set_xlabel('steps')
            self.action_ploter.ax[0].set_ylabel('throttle')
            #accelerate
                      
            ##brake
            self.action_ploter.ax[1].clear()
            self.action_ploter.ax[1].plot(agent.brake_controller,'r',linewidth=0.5,label='brake controller')
            self.action_ploter.ax[1].plot(agent.brake_imitation,'g',linewidth=0.5,label='brake imitation')
            self.action_ploter.ax[1].legend(loc='lower left')
            self.action_ploter.ax[1].set_xlabel('steps')
            self.action_ploter.ax[1].set_ylabel('brake')
            ##brake

            #steering
            self.action_ploter.ax[2].clear()
            self.action_ploter.ax[2].plot(agent.steering_angle_controller,'r',linewidth=0.5,label='steering controller')
            self.action_ploter.ax[2].plot(agent.steering_angle_imitation,'g',linewidth=0.5,label='steering imitation')
            self.action_ploter.ax[2].plot(agent.diff_steering,'b',linewidth=0.5,label='steering difference')
            self.action_ploter.ax[2].legend(loc='lower left')
            self.action_ploter.ax[2].set_xlabel('steps')
            self.action_ploter.ax[2].set_ylabel('steering')
            #steering
    
            self.action_ploter.fig = plt.gcf() 
            self.action_ploter.fig.canvas.draw() 
            self.action_ploter.fig.canvas.flush_events() 
        
        #speed   
        if hasattr(self, 'speed_ploter'):
            
            self.speed_ploter.ax[0].clear()
            self.speed_ploter.ax[0].plot(agent.speed,'r',linewidth=0.5,label='speed')
            self.speed_ploter.ax[0].plot(agent.set_point_speed,'g',linewidth=0.5,label='speed set point')
            self.speed_ploter.ax[0].plot(agent.optic_speed,'b',linewidth=0.5,label='optic speed')
            self.speed_ploter.ax[0].legend(loc='lower left')
            self.speed_ploter.ax[0].set_xlabel('steps')
            self.speed_ploter.ax[0].set_ylabel('speed m/s')
    
            self.speed_ploter.fig = plt.gcf() 
            self.speed_ploter.fig.canvas.draw() 
            self.speed_ploter.fig.canvas.flush_events() 
            
        if hasattr(self, 'time_ploter'):
                
            self.time_ploter.ax[0].clear()
            self.time_ploter.ax[0].plot(agent.time_step_set,'g',linewidth=0.5,label='time step')
            self.time_ploter.ax[0].legend(loc='lower left')
            self.time_ploter.ax[0].set_xlabel('steps')
            self.time_ploter.ax[0].set_ylabel('time step s')
            
            self.time_ploter.fig = plt.gcf() 
            self.time_ploter.fig.canvas.draw() 
            self.time_ploter.fig.canvas.flush_events() 