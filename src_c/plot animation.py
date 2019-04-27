#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 23:08:59 2019

@author: spikezz
"""

#from math import sin, cos
#import numpy as np
#from scipy.integrate import odeint
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#
#t = np.arange(0, 20, 0.1)
##track = odeint(pendulum_equations1, (1.0, 0), t, args=(leng,))
#track = odeint(pendulum_equations2, (1.0, 0), t, args=(leng, b_const))
#xdata = [leng*sin(track[i, 0]) for i in range(len(track))]
#ydata = [-leng*cos(track[i, 0]) for i in range(len(track))]
#
#
#fig, ax = plt.subplots()
#ax.grid()
#line, = ax.plot([], [], 'o-', lw=2)
#time_template = 'time = %.1fs'
#time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
#
#
#def init():
#    ax.set_xlim(-20, 20)
#    ax.set_ylim(-20, 20)
#    time_text.set_text('')
#    return line, time_text
#
#def update(i):
#    newx = [0, xdata[i]]
#    newy = [0, ydata[i]]
#    line.set_data(newx, newy)
#    time_text.set_text(time_template %(0.1*i))
#    return line, time_text

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as line

fig = plt.figure()
ax = plt.axes(xlim=(0,10), ylim=(0,1))
linem = line.Line2D([],[])
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    ax.add_line(linem)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    time_text.set_text('')
    return linem,time_text

def update(i):
#    newx1 = [i, 2*i]
#    newy1 = [i, ydata[i]]
#    newx2 = [i, -xdata[i]]
#    newy2 = [i, ydata[i]]
#    line1.set_data(newx1, newy1)
#    line2.set_data(newx2, newy2)
#    print("i:",i)
#    linem.set_xdata(np.arange(0,10,1))
#    linem.set_ydata(np.random.rand(10))
    
    return linem,time_text

fig.show()

ani = animation.FuncAnimation(fig, update,
                              init_func=init,
                              frames=1,
                              interval=50,
                              blit=True)

#raw_input("wait")