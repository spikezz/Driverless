#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:24:24 2019

@author: spikezz
"""

import numpy as np

class Initializer(object):
    
    def __init__(self,car_state):
        
        self.initial_velocoty_noise=self.calibration_speed_sensor(car_state)    

    def calibration_speed_sensor(self,car_state):
        
        init_v=np.zeros(3)
        
        #calibration white noise of velocity
        init_v[0]=car_state.kinematics_estimated.linear_velocity.x_val
        init_v[1]=car_state.kinematics_estimated.linear_velocity.y_val
        init_v[2]=car_state.kinematics_estimated.linear_velocity.z_val
        #calibration white noise of velocity
    
        return init_v