#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:46:19 2019

@author: spikezz
"""

import signal
from xbox360controller import Xbox360Controller

def on_button_pressed(button):
    print('Button {0} was pressed'.format(button.name))


def on_button_released(button):
    print('Button {0} was released'.format(button.name))


def on_thumbstick_axis_moved(axis):
    print('Axis {0} moved to {1} {2}'.format(axis.name, axis.x, axis.y))
    
def on_trigger_axis_moved(axis):
    print('Axis {0} moved to {1}'.format(axis.name, axis._value))

try:
    with Xbox360Controller(0, axis_threshold=0.2) as controller:
        # Button A events
        controller.button_a.when_pressed = on_button_pressed
        controller.button_a.when_released = on_button_released
        
        controller.trigger_l.when_moved = on_trigger_axis_moved
        controller.trigger_l.when_moved = on_trigger_axis_moved
        
        controller.trigger_r.when_moved = on_trigger_axis_moved
        controller.trigger_r.when_moved = on_trigger_axis_moved
        
        controller.button_start.when_pressed = on_button_pressed
        controller.button_start.when_released = on_button_released

        # Left and right axis move event
        controller.axis_l.when_moved = on_thumbstick_axis_moved
        controller.axis_r.when_moved = on_thumbstick_axis_moved

        signal.pause()
        
except KeyboardInterrupt:
    pass