#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 22:55:00 2019

@author: spikezz
"""

from scipy.special import fresnel
import matplotlib.pyplot as plt
import calculate as cal
import math


def find_circle(x_test,y_test,i):

 
    a_f=(x_test[i-1]**2-x_test[i]**2)
    b_f=(x_test[i]-x_test[i-1])
    c_f=(y_test[i-1]**2-y_test[i]**2)
    d_f=(y_test[i]-y_test[i-1])
    
    e_f=(x_test[i]**2-x_test[i+1]**2)
    f_f=(x_test[i+1]-x_test[i])
    g_f=(y_test[i]**2-y_test[i+1]**2)
    h_f=(y_test[i+1]-y_test[i])
    
    A=2*b_f
    B=2*d_f
    C=a_f+c_f
    
    D=2*f_f
    E=2*h_f
    F=e_f+g_f
    
    y_center=(D*C-A*F)/(A*E-B*D)
    x_center=(B*F-C*E)/(A*E-B*D)
    ax.plot(x_center,y_center,'bo',markersize=2)
    r=pow(((x_center-x_test[i])**2+(y_center-y_test[i])**2),0.5)
    
    norm_vector=[x_test[i]-x_center,y_test[i]-y_center]
    
    return x_center,y_center,r,norm_vector

def draw_circle(x_center,y_center,r):
    
    x_rs=[]
    y_rs=[]
    theta=0
    
    for theta in range(360):
        x_r=math.cos(math.radians(theta))*r+x_center
        y_r=math.sin(math.radians(theta))*r+y_center
        x_rs.append(x_r)
        y_rs.append(y_r)
        
    ax.plot(x_rs,y_rs,'b',linewidth=1)

def draw_kurve(curvstart, curvend, p_this, p_next,arc_length,arc_theta, step, color, line_width, ax):
    
    factor_scale=5
    arc_a=0
    while arc_length-arc_a>0:
#        ax.lines.clear()
        print("factor_scale:",factor_scale)
        xs = []
        ys = []
        rs = []
        ns = []
        n_start=curvstart*factor_scale/math.pi
        n_end=curvend*factor_scale/math.pi
        print("n_start:",n_start)
        print("n_end:",n_end)
        n=n_start
        
        if curvstart<curvend and curvstart>0 and curvend>0:
            
            while (n<n_end+step):
                
                x, y =fresnel(n)
                xs.append(x)
                ys.append(y)
                rs.append(math.pi/n)
                ns.append(n)
                n+=step
                
            print("end step:",n-step) 
#            print("xs:",xs)
#            print("ys:",ys)
            print("point_start:",[xs[0],ys[0]])
            print("point_end:",[xs[-1],ys[-1]])
            
            arc_b=cal.calculate_radius([xs[0],ys[0]],[xs[-1],ys[-1]])
            print("arc before scale:",arc_b)
#            scale_rate=arc_length/arc
#            print("scale_rate:",scale_rate)
    
            xs=list(map((lambda x: x *factor_scale), xs))
            ys=list(map((lambda x: x *factor_scale), ys))
            
            arc_a=cal.calculate_radius([xs[0],ys[0]],[xs[-1],ys[-1]])
            print("arc after scale:",arc_a)
            
#     
#            ax.plot([xs[0],xs[-1]],[ys[0],ys[-1]],color,linewidth=line_width)
#            ax.plot(xs,ys,color,linewidth=line_width)
            
        elif curvstart>curvend and curvstart>0 and curvend>0:
            pass
        
        factor_scale+=0.001     
    
    curve_theta=cal.calculate_sita_of_radius([0,0],[xs[-1]-xs[0],ys[-1]-ys[0]])%360
    print("curve vector:",curve_theta)
    delta_theta=arc_theta-curve_theta
    print("delta_theta:",delta_theta)
    for i in range(len(xs)):
        
        radius=cal.calculate_radius([0,0],[xs[i],ys[i]])
        angle=cal.calculate_sita_of_radius([0,0],[xs[i],ys[i]])
#        print("radius:",radius)  
#        print("angle:",angle)  
        rotated_point=cal.calculate_rotated_point(delta_theta,radius,angle)
        xs[i]=rotated_point[0]
        ys[i]=rotated_point[1]

    shift_vector=[p_this[0]-xs[0],p_this[1]-ys[0]]
    print("shift_vector:",shift_vector)
        
    xs=list(map((lambda x: x +shift_vector[0]), xs))
    ys=list(map((lambda x: x +shift_vector[1]), ys))  
    
    ax.plot([xs[0],xs[-1]],[ys[0],ys[-1]],color,linewidth=line_width)
    ax.plot(xs,ys,color,linewidth=line_width)
    
fig = plt.figure()
fig.show()
ax = fig.add_subplot(1, 1, 1)
fig.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9)
#ax.set_xlim(-0, 20)
#ax.set_ylim(-10, 10)
step=0.01

x_test=[0,0,5,6]
y_test=[0,5,10,9]

x_c=[0]*(len(x_test)-2)
y_c=[0]*(len(y_test)-2)
r_c=[0]*(len(x_test)-2)
k_c=[0]*(len(x_test)-2)
n_c=[0]*(len(x_test)-2)

ax.plot(x_test,y_test,'g',linewidth=1)


for i in range(1,(len(x_test)-1)):

    x_c[i-1],y_c[i-1],r_c[i-1],n_c[i-1]=find_circle(x_test,y_test,i)
    k_c[i-1]=1/r_c[i-1]
    draw_circle(x_c[i-1],y_c[i-1],r_c[i-1])
    print("circle center:",x_c[i-1],y_c[i-1],r_c[i-1])

print("r_c:",r_c)    
print("k_c:",k_c)

i=1
p_this=[x_test[i],y_test[i]]
p_next=[x_test[i+1],y_test[i+1]]
print("p_this:",p_this)
print("p_next:",p_next)
#p_start,p_end,arc_length=
arc_length=cal.calculate_radius(p_this,p_next)
print("arc_length:",arc_length)
arc_theta=cal.calculate_sita_of_radius([0,0],[p_next[0]-p_this[0],p_next[1]-p_this[1]])
print("arc vector:",arc_theta)
draw_kurve(k_c[i-1],k_c[i],p_this,p_next,arc_length,arc_theta,step,'r',1,ax)

#draw_kurve(1/40.,1/2.,p_this,p_next,arc_length,step,'r',1,ax)
#p_start,p_end,arc_length=draw_kurve(k_c[i-1],k_c[i],p_this,p_next,step,j,xs,ys,rs,ns,'r',1,ax)