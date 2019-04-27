#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:59:09 2019

@author: spikezz
"""
from __future__ import division
import math
#import warnings
import numpy

# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# helper functions

def calculate_radius(point_0,point_1):
    
    R=math.sqrt(pow(point_1[0]-point_0[0],2)+pow(point_1[1]-point_0[1],2))
    
    return R

def calculate_sita_of_radius(point_0,point_1):

    if point_1[0]-point_0[0] <0:

        sita=180+math.degrees(math.atan((point_1[1]-point_0[1])/(point_1[0]-point_0[0])))           
    
    elif point_1[0]-point_0[0] >0:
        
        sita=math.degrees(math.atan((point_1[1]-point_0[1])/(point_1[0]-point_0[0])))
        
    else:
        
        if point_1[1]-point_0[1]>=0:
            
            sita=90
            
        else:
            
            sita=-90
            
    if sita >=180 and sita<270:
        
        sita=sita-360

    return sita

def calculate_t(line,colour,distance):
    
    tpoint=[0,0]
    sita_l=0
    sita_t=0
    vek_l=[line[1][0]-line[0][0],line[1][1]-line[0][1]]
    
    if vek_l[0]!=0:
            
        sita_l=calculate_sita_of_radius(vek_l,[0,0])

    else:

            
        if vek_l[1]>0:
            
            sita_l=90
        
        else:
            
            sita_l=-90
                
    if colour==1:
        
        sita_t=sita_l-90
        
    else:
        
        sita_t=sita_l+90
    
    tpoint[0]=math.cos(math.radians(sita_t))*distance+vek_l[0]+line[0][0]
    tpoint[1]=math.sin(math.radians(sita_t))*distance+vek_l[1]+line[0][1]
    
    return tpoint

def calculate_projection(switch,short,long,bottom):
    
    projektion=[0,0]
    
    if switch==0:
#        angle=math.degrees(math.acos((pow(short,2)+pow(bottom,2)-pow(long,2))/(2*short*bottom)))
        angle=math.acos((pow(short,2)+pow(bottom,2)-pow(long,2))/(2*short*bottom))
#        print("angle",angle)
        projektion[0]=short*math.cos(angle)
        projektion[1]=short*math.sin(angle)
#        print("projektion[1]",projektion[1])

    else:
#        angle=math.degrees(math.acos((pow(long,2)+pow(bottom,2)-pow(short,2))/(2*long*bottom)))
        angle=math.acos((pow(long,2)+pow(bottom,2)-pow(short,2))/(2*long*bottom))
#        print("angle",angle)
        
        projektion[0]=long*math.cos(angle)
        projektion[1]=long*math.sin(angle)
#        print("projektion[1]",projektion[1])
#    short*math.cos(math.radians(angle))
    return projektion

def calculate_rotated_point(theta,R,sita):
    
    point=[R*math.cos(math.radians(theta+sita)),R*math.sin(math.radians(theta+sita))]
    
    return point

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)
