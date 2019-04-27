#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 22:25:34 2019

@author: spikezz
"""

#import numpy as np
#import pylab as pl
#from scipy import interpolate 
import matplotlib.pyplot as plt
from math import cos, sin, pi, radians, sqrt
from scipy.special import fresnel



#x = np.linspace(0, 2*np.pi+np.pi/4, 10)
#y = np.sin(x)
#
#x_new = np.linspace(0, 2*np.pi+np.pi/4, 100)
#f_linear = interpolate.interp1d(x, y)
#tck = interpolate.splrep(x, y)
#y_bspline = interpolate.splev(x_new, tck)
#
#plt.xlabel(u'/A')
#plt.ylabel(u'/V')
#
#plt.plot(x, y, "o",  label=u"ori")
#plt.plot(x_new, f_linear(x_new), label=u"l_ip")
#plt.plot(x_new, y_bspline, label=u"B-spline")
#
#pl.legend()
#pl.show()


def spiral_interp_centre(distance, x, y, hdg, length, curvstart,curvEnd,factor):
    '''Interpolate for a spiral centred on the origin'''
    # s doesn't seem to be needed...
    theta = hdg                    # Angle of the start of the curve
    Ltot = length                  # Length of curve
    Rend = 1 / curvEnd             # Radius of curvature at end of spiral

    if curvstart!=0:
        
        Rstart = 1 / curvstart
    # Rescale, compute and unscale
#    a = 1 / sqrt(2 * Ltot * Rend)  # Scale factor
    a = 1/pow(factor* Ltot * Rend,0.5)
    distance_scaled = distance * a # Distance along normalised spiral
    print("distance_scaled:%.2f"%(distance_scaled))

    deltay_scaled, deltax_scaled = fresnel(distance_scaled)
#    print("deltax_scaled:%.2f"%(deltax_scaled))
#    print("deltay_scaled:%.2f"%(deltay_scaled))
    deltax = deltax_scaled / a
    deltay = deltay_scaled / a

    # deltax and deltay give coordinates for theta=0
    deltax_rot = deltax * cos(theta) - deltay * sin(theta)
    deltay_rot = deltax * sin(theta) + deltay * cos(theta)

    # Spiral is relative to the starting coordinates
    xcoord = x + deltax_rot
    ycoord = y + deltay_rot

    return xcoord, ycoord , distance_scaled
#    return deltax_scaled, deltay_scaled

def draw(kurve_length,curvstart,curvEnd,factor):
      
    xs = []
    ys = []
    ns = []
    ds = []
    
    kurve_length=1000
    curvstart=1/10000000.
    curvEnd=1/20.
    factor=4
    start_point=(curvstart/curvEnd)*kurve_length
    print("start_point:",start_point)
    af=1/pow(factor* kurve_length / curvEnd,0.5)
    print("factor:%.12f"%(af))
    
    #for factor in range(2, 5, 2):
    #    print("factor:",factor)
    #for n in range(start_point, kurve_length+1,0.1):
    n=start_point
    while (n<kurve_length+0.1):
        
        x, y,d = spiral_interp_centre(n, 0, 0, radians(0), kurve_length, curvstart ,curvEnd,factor)
        if n==start_point:
            print("start point:",x, y)
        if n==kurve_length:
            print("end point:",x, y)
            
        xs.append(x)
        ys.append(-y)
        ns.append(n)
        ds.append(d)
        print("vor",n)
        n+=0.1
        print("nach",n)
    ax.plot(xs, ys)
    ax2.plot(ns, ds)
    ax3.plot(ns,xs)
    ax4.plot(ns,ys)

fig = plt.figure()
fig.show()
ax = fig.add_subplot(1, 1, 1)
fig.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9)
#ax.set_xlim(-100, 100)
#ax.set_ylim(-100, 100)
ax.set_xlim(0, 300)
ax.set_ylim(-300, 0)
#plt.ion()
# This version
fig2 = plt.figure()
fig2.show()
ax2 = fig2.add_subplot(2, 2, 1)
ax3 = fig2.add_subplot(2, 2, 2)
ax4 = fig2.add_subplot(2, 2, 3)
ax5 = fig2.add_subplot(2, 2, 4)
fig2.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9)
#ax2.set_xlim(-100, 100)
#ax2.set_ylim(-100, 100)

draw()

## Your version
#from yourspiral import spiralInterpolation
#stations = spiralInterpolation(1, 77, 50, 100, radians(56), 40, 0, 1/20.)
#ax.plot(stations[:,0], stations[:,1])
#
#ax.legend(['My spiral', 'Your spiral'])
#fig.savefig('spiral.png')
#plt.show()



#t = np.linspace(0, 5.0, 201)
#ss, cc = fresnel(t / np.sqrt(np.pi / 2))
#scaled_ss = np.sqrt(np.pi / 2) * ss
#scaled_cc = np.sqrt(np.pi / 2) * cc
#plt.plot(t, scaled_cc, 'g--', t, scaled_ss, 'r-', linewidth=2)
#plt.grid(True)
#plt.show()