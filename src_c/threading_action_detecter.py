#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:39:57 2019

@author: spikezz
"""

#!/usr/bin/python3

import threading
import signal

from xbox360controller import Xbox360Controller


class action_detecter (threading.Thread):
    
    def __init__(self, threadID, name):
        
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.throttle_signal=0
        self.brake_signal=0
        self.steering_signal=0
        
    def on_button_pressed(self,button):
        
        print('Button {0} was pressed'.format(button.name))
    
    def on_button_released(self,button):
        
        print('Button {0} was released'.format(button.name))
    
    def on_thumbstick_axis_moved(self,axis):
        
        print('Axis {0} moved to {1} {2}'.format(axis.name, axis.x, axis.y))
        
        if axis.name=='axis_l':
            
            self.steering_signal=axis.x
            
    def on_trigger_axis_moved(self,axis):
        
        print('Axis {0} moved to {1}'.format(axis.name, axis._value))
        
        if axis.name=='trigger_l':

            self.brake_signal=axis._value
    
        if axis.name=='trigger_r': 

            self.throttle_signal=axis._value
            
    def run(self):
            
        print ("Threading start" + self.name)
        try:
        
            with Xbox360Controller(0, axis_threshold=0.2) as controller:
                # Button A events
                controller.button_a.when_pressed = self.on_button_pressed
                controller.button_a.when_released = self.on_button_released
                
                controller.trigger_l.when_moved = self.on_trigger_axis_moved
        #            controller.trigger_l.when_moved = on_trigger_axis_moved
                
                controller.trigger_r.when_moved = self.on_trigger_axis_moved
        #            controller.trigger_r.when_moved = on_trigger_axis_moved
                
                controller.button_start.when_pressed = self.on_button_pressed
                controller.button_start.when_released = self.on_button_released
        
                # Left and right axis move event
                controller.axis_l.when_moved = self.on_thumbstick_axis_moved
                controller.axis_r.when_moved = self.on_thumbstick_axis_moved
                
                print('start')
                signal.pause()
            
        except KeyboardInterrupt:
            
            print('end')


#def print_time(threadName, delay, counter):
#    while counter:
#        if exitFlag:
#            threadName.exit()
#        time.sleep(delay)
#        print ("%s: %s" % (threadName, time.ctime(time.time())))
#        counter -= 1

## 创建新线程
#thread1 = action_detecter(1, "Thread-1", 1)
#thread2 = action_detecter(2, "Thread-2", 2)
#
## 开启新线程
#thread1.start()
#thread2.start()
#thread1.join()
#thread2.join()
#print ("退出主线程")