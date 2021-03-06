#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:20:46 2019

@author: spikezz
"""
import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt
import csv

class Saver(object):
    
    def __init__(self,sess,load,actor,critic,all_var,agent_i=False, agent_r=False):
        
        List_net=[]
        List_ae=[v for v in actor.e_params]
        List_net.extend(List_ae)
        
        if agent_r:
            
            List_at=[v for v in actor.t_params]
            List_ct=[v for v in critic.t_params]
            List_ce=[v for v in critic.e_params]
            List_net.extend(List_at)
            List_net.extend(List_ct)
            List_net.extend(List_ce)
        
        if all_var==True:
            
            self.saver=tf.train.Saver(max_to_keep=10000)
            
        else:
            
            self.saver=tf.train.Saver(var_list=List_net,max_to_keep=10000)
            
        self.LOAD = load
        self.MODE = ['0']
        self.n_model = 0
#        self.di = './Model/Model_'+self.MODE[self.n_model]
#        di_load = './Model/Model_0'
        
        self.di = '/media/spikezz/1E50709050706FFF/Driverless/Model/Model_'+self.MODE[self.n_model]
        di_load = '/media/spikezz/1E50709050706FFF/Driverless/Model/Model_0'
#        self.di = '/media/spikezz/1E50709050706FFF/Driverless/Model/Model_'+self.MODE[self.n_model]
#        di_load = '/media/spikezz/1E50709050706FFF/Driverless/Model/Model_0'
        if self.LOAD:
            
            sess.run(tf.global_variables_initializer())
    
            self.saver.restore(sess, tf.train.latest_checkpoint(di_load))
        else:
            
            if os.path.isdir(di_load): shutil.rmtree(di_load)
            sess.run(tf.global_variables_initializer())
            os.mkdir(di_load)
    
    def save(self,sess,running_reward,reward_ep_mean):
        
        self.n_model+=1
        self.MODE.append(str(self.n_model))
        
        if os.path.isdir(self.di): shutil.rmtree(self.di)
        os.mkdir(self.di)

        ckpt_path = os.path.join( '/media/spikezz/1E50709050706FFF/Driverless/Model/Model_'+self.MODE[self.n_model], 'DDPG.ckpt')
#        ckpt_path = os.path.join( '/media/spikezz/1E50709050706FFF/Driverless/Model/Model_'+self.MODE[self.n_model], 'DDPG.ckpt')
        save_path = self.saver.save(sess, ckpt_path, write_meta_graph=False)
        print("\nSave Model %s\n" % save_path)

#        file = os.path.join( '/media/spikezz/1E50709050706FFF/Driverless/Model/Model_'+self.MODE[self.n_model], 'episode_reward.txt')
        file = os.path.join( '/media/spikezz/1E50709050706FFF/Driverless/Model/Model_'+self.MODE[self.n_model], 'episode_reward.txt')
        fw=open(file, mode='w')
        reward_str= 'running_reward:'+str(running_reward)+'\n'+'reward_ep_mean:'+str(reward_ep_mean)
        fw.seek(0,0)
        fw.write( reward_str)

#        current_path = os.getcwd()
#        model_dir = os.path.join(current_path, 'logs')

class Ploter(object):
    
    def __init__(self,xlim,ylim,fig_size):
        
        self.fig = plt.figure(figsize=fig_size)
        self.fig.show()
        self.ax = self.fig.add_subplot(1,1,1)
        self.fig.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9)
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])
    
        plt.ion()
    
class Summary(object):
    
    def writer_data(self,agent,odm_msg_set,eul_msg_set,time_set):
  
        with open('motor_model.csv', 'w', newline='') as m:
            
            writer_motor = csv.writer(m)
            
            for i in range(0,len(time_set)):
                
                input_output_motor=[agent.actor.throttle_controller[i],agent.actor.speed[i]]
                
                writer_motor.writerow(input_output_motor)
                
     
        with open('auto_dynamic.csv', 'w', newline='') as f:
            
            writer_auto_dynamic = csv.writer(f)
            for i in range(0,len(time_set)):
                
                action_state_turple=[time_set[i],agent.actor.throttle_controller[i],agent.actor.brake_controller[i],\
                                     agent.actor.steering_angle_controller[i],odm_msg_set[i].header.stamp.secs,\
                                     odm_msg_set[i].header.stamp.nsecs,odm_msg_set[i].pose.pose.position.x,\
                                     odm_msg_set[i].pose.pose.position.y,eul_msg_set[i].z,odm_msg_set[i].twist.twist.linear.x,\
                                     odm_msg_set[i].twist.twist.linear.y,odm_msg_set[i].twist.twist.angular.z]                                
                
                writer_auto_dynamic.writerow(action_state_turple)
                
        
#    def plot_summary(self,LOAD,var,pointer,capacity,reward_ep,running_reward,
#                running_reward_max,reward_mean,rr_idx,max_reward_reset,
#                action_ori,actor,reward_one_ep_mean,rr,critic,reward_mean_max_rate,ep_lr,time_set):
#        
#        print("LOAD:",LOAD)
#        print("critic.rank_TD_max:",critic.rank_TD_max,"critic.rank_TD_min:",critic.rank_TD_min)
##        print("FPS:",clock.get_fps())
#        print("var0:",var[0],"var1:",var[1],"var2:",var[2])
#        print("MEMORY_pointer:",pointer)
#        print("MEMORY_CAPACITY:",capacity)
#        print("running_reward:",running_reward)
#        print("max_running_reward:",running_reward_max)
#        print("reward_mean:",reward_mean[rr_idx-1])
#        print("max_reward_reset:",max_reward_reset)
#        print("lr ep :",ep_lr)
         #accelerate 
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
                
        def __init__(self,plote_action=False,plote_speed=False,plote_cross_position=False,plote_predict_angle=False,\
                                    plote_predict_curverature=False):
            
            if plote_action:
                
                self.action_ploter=self.Summary_Ploter(row=1, column=3)
                
            if plote_speed:
                
                self.speed_ploter=self.Summary_Ploter(row=1, column=1)
                
            if plote_cross_position:
                
                self.cross_position_ploter=self.Summary_Ploter(row=1, column=1)
                
            if plote_predict_angle:
                
                self.predict_angle_ploter=self.Summary_Ploter(row=1, column=1)
                
            if plote_predict_curverature:
                
                self.predict_curverature_ploter=self.Summary_Ploter(row=1, column=1)
            
 
    
    #            writer.writerows(agent)
        def plot_summary(self,agent):
            
            #accelerate
            if hasattr(self, 'action_ploter'):
                
                self.action_ploter.ax[0].plot(agent.actor.throttle_controller,'r',linewidth=0.5)
                self.action_ploter.ax[0].plot(agent.actor.throttle_imitation,'g',linewidth=0.5)
                self.action_ploter.ax[0].set_xlabel('steps')
                self.action_ploter.ax[0].set_ylabel('throttle_controller-red/throttle_imitation-green')
                #accelerate
        
                ##brake
                self.action_ploter.ax[1].plot(agent.actor.brake_controller,'r')
                self.action_ploter.ax[1].plot(agent.actor.brake_imitation,'g',linewidth=0.5)
                self.action_ploter.ax[1].set_xlabel('steps')
                self.action_ploter.ax[1].set_ylabel('brake_controller-red/brake_imitation-green')
                ##brake
        
                #steering
                self.action_ploter.ax[2].plot(agent.actor.steering_angle_controller,'r')
                self.action_ploter.ax[2].plot(agent.actor.steering_angle_imitation,'g',linewidth=0.5)
                self.action_ploter.ax[2].set_xlabel('steps')
                self.action_ploter.ax[2].set_ylabel('steering_angle_controller-red/steering_angle_imitation-green')
                #steering
        
                self.action_ploter.fig = plt.gcf() 
                self.action_ploter.fig.canvas.draw() 
                self.action_ploter.fig.canvas.flush_events() 
            
            #speed   
            if hasattr(self, 'speed_ploter'):
                    
                self.speed_ploter.ax[0].plot(agent.actor.speed,'r',linewidth=0.5)
                self.speed_ploter.ax[0].plot(agent.actor.set_point_speed,'g',linewidth=0.5)
                self.speed_ploter.ax[0].set_xlabel('steps')
                self.speed_ploter.ax[0].set_ylabel('speed-red/set_point_speed-green')
        
                self.speed_ploter.fig = plt.gcf() 
                self.speed_ploter.fig.canvas.draw() 
                self.speed_ploter.fig.canvas.flush_events() 
            
            #cross position  
            if hasattr(self, 'cross_position_ploter'):
                    
                self.cross_position_ploter.ax[0].plot(agent.actor.lateral_position,'r',linewidth=0.5)
                self.cross_position_ploter.ax[0].plot(agent.actor.set_point_lateral_position,'g',linewidth=0.5)
                self.cross_position_ploter.ax[0].set_xlabel('steps')
                self.cross_position_ploter.ax[0].set_ylabel('cross_position-red/set_point_cross_position-green')
        
                self.cross_position_ploter.fig = plt.gcf() 
                self.cross_position_ploter.fig.canvas.draw() 
                self.cross_position_ploter.fig.canvas.flush_events() 
            
            #predict angle measurement
            if hasattr(self, 'predict_angle_ploter'):
                
                self.predict_angle_ploter.ax[0].plot(agent.actor.predict_angle_diffrence,'r',linewidth=0.5)
                self.predict_angle_ploter.ax[0].set_xlabel('steps')
                self.predict_angle_ploter.ax[0].set_ylabel('predict_angle_diffrence-red')
        
                self.predict_angle_ploter.fig = plt.gcf() 
                self.predict_angle_ploter.fig.canvas.draw() 
                self.predict_angle_ploter.fig.canvas.flush_events() 
            
            #predict curverature measurement
            if hasattr(self, 'predict_curverature_ploter'):
                
#                self.predict_curverature_ploter.ax[0].plot(agent.actor.predict_curverature_measured,'r',linewidth=0.5)
                self.predict_curverature_ploter.ax[0].plot(agent.actor.predict_curverature,'g',linewidth=0.5)
                self.predict_curverature_ploter.ax[0].plot(agent.actor.open_close_control_rate,'b',linewidth=0.5)
                self.predict_curverature_ploter.ax[0].set_xlabel('steps')
                self.predict_curverature_ploter.ax[0].set_ylabel('predict_curverature-red')
                
                self.predict_curverature_ploter.fig = plt.gcf() 
                self.predict_curverature_ploter.fig.canvas.draw() 
                self.predict_curverature_ploter.fig.canvas.flush_events() 
        
                
    #        #probability
    #        plt.figure()
    #        plt.subplot(311)
    #        plt.plot(action_ori[3])  
    #        plt.xlabel('steps')
    #        plt.ylabel('p accelerate')
    #        plt.subplot(312)
    #        plt.plot(action_ori[4])  
    #        plt.xlabel('steps')
    #        plt.ylabel('p brake')
    #        plt.subplot(313)
    #        plt.plot(action_ori[5])  
    #        plt.xlabel('steps')
    #        plt.ylabel('p idle')
    #        #probability
    #        
    #        
    #        #reward
    #        plt.figure()
    #        plt.subplot(211)
    #        plt.plot(reward_ep)  
    #        plt.xlabel('steps')
    #        plt.ylabel('reward one ep')
    #        plt.subplot(212)
    #        plt.plot(reward_one_ep_mean)  
    #        plt.xlabel('episode steps')
    #        plt.ylabel('reward mean for one ep')
    #        plt.figure()
    #        plt.subplot(311)
    #        plt.plot(rr)  
    #        plt.xlabel('episode steps')
    #        plt.ylabel('runing reward whole episode')
    #        plt.subplot(312)
    #        plt.plot(reward_mean)  
    #        plt.xlabel('episode steps')
    #        plt.ylabel('reward_mean')
    #        plt.subplot(313)
    #        plt.plot(reward_mean_max_rate)  
    #        plt.xlabel('episode steps')
    #        plt.ylabel('reward Max/mean')
    #        
    #        plt.figure()
    #        plt.subplot(211)
    #        plt.plot(critic.model_localization)
    #        plt.xlabel('learning steps')
    #        plt.ylabel('model_localization')
    #        plt.subplot(212)
    #        plt.plot(time_set)
    #        plt.xlabel('cycle steps')
    #        plt.ylabel('time_set')
    #
    #        
    #        plt.show()
    #        
