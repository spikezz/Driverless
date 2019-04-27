#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:54:19 2019

@author: spikezz
"""

from scipy.special import fresnel
import matplotlib.pyplot as plt
import calculate as cal
import math
    
def draw_kurve(curvstart, curvend, p_this, p_next, step, i, xs, ys, rs, ns, color, line_width, ax):
    print("rstart:",1/curvstart)
    print("rnd:",1/curvend)
#    n=step
    n=curvstart*math.pi
    n_end=curvend*math.pi
    length=cal.calculate_radius(p_this,p_next)
    print("arc length:",length)
    print("n:",n)
    print("n_end:",n_end)
    
    if curvstart<curvend and curvstart>0 and curvend>0:
        
        while (n<n_end+step):
            
            print("n:",n)
            y, x = fresnel(n*factor_speed)
            y=y*(1)
            xs[i].append(x/factor_scale)
            ys[i].append(y/factor_scale)
            rs[i].append(math.pi/n)
            ns[i].append(n)
            
            if n==curvstart*math.pi:
                
                x_start=x/factor_scale
                y_start=y/factor_scale
                
            elif n+step>=n_end+step:
                
                x_end=x/factor_scale
                y_end=y/factor_scale
                
            n+=step
            

        arc=cal.calculate_radius([x_start,y_start],[x_end,y_end])
        print("arc before:",arc)
        scale_rate=length/arc
        x_start=x_start*scale_rate
        y_start=y_start*scale_rate
        x_end=x_end*scale_rate
        y_end=y_end*scale_rate
        print("p_start:",[x_start,y_start])
        print("p_end:",[x_end,y_end])
        print("scale_rate:",scale_rate)
        print("p_this:",p_this)
        print("p_next:",p_next)
        shift_vector=[p_this[0]-x_start,p_this[1]-y_start]
        print("shift_vector:",shift_vector)
    #    print("xs[i]:",xs[i])
    #    print("ys[i]:",ys[i])
#        xs[i]=list(map((lambda x: x *scale_rate+shift_vector[0]), xs[i]))
#        ys[i]=list(map((lambda x: x *scale_rate+shift_vector[1]), ys[i]))
        xs[i]=list(map((lambda x: x *scale_rate), xs[i]))
        ys[i]=list(map((lambda x: x *scale_rate), ys[i]))
        print("end step:",n-step)   
        print("end point:",x, y)
#        x_start=x_start+shift_vector[0]
#        y_start=y_start+shift_vector[1]
#        x_end=x_end+shift_vector[0]
#        y_end=y_end+shift_vector[1]
        arc=cal.calculate_radius([x_start,y_start],[x_end,y_end])
        print("arc after:",arc)
        ax.plot([x_start,y_start],[x_end,y_end],color,linewidth=line_width)
        ax.plot(xs[i],ys[i],color,linewidth=line_width)
    #    ax.plot(ns[i],rs[i],color,linewidth=line_width)
        return [x_start,y_start],[x_end,y_end],arc_length
        
    elif curvstart>curvend and curvstart>0 and curvend>0:
        
        n=n*(-1)
        n_end=n_end*(-1)
        while (n<n_end+step):
            y, x = fresnel(n*factor_speed)
            y=y*(1)
            xs[i].append(x/factor_scale)
            ys[i].append(y/factor_scale)
            rs[i].append(math.pi/n)
            ns[i].append(n)
            if n==-curvstart*math.pi:
                
                x_start=x/factor_scale
                y_start=y/factor_scale
                
            elif n+step>=n_end+step:
                
                x_end=x/factor_scale
                y_end=y/factor_scale
                
            n+=step
            
        arc=cal.calculate_radius([x_start,y_start],[x_end,y_end])
        scale_rate=length/arc
        x_start=x_start*scale_rate
        y_start=y_start*scale_rate
        x_end=x_end*scale_rate
        y_end=y_end*scale_rate
        print("p_start:",[x_start,y_start])
        print("p_end:",[x_end,y_end])
        print("scale_rate:",scale_rate)
        print("p_this:",p_this)
        print("p_next:",p_next)
        shift_vector=[p_this[0]-x_start,p_this[1]-y_start]
        print("shift_vector:",shift_vector)
        
    #    print("xs[i]:",xs[i])
    #    print("ys[i]:",ys[i])
#        xs[i]=list(map((lambda x: x *scale_rate+shift_vector[0]), xs[i]))
#        ys[i]=list(map((lambda x: x *scale_rate+shift_vector[1]), ys[i]))
        xs[i]=list(map((lambda x: x *scale_rate), xs[i]))
        ys[i]=list(map((lambda x: x *scale_rate), ys[i]))

        
        print("end step:",n-step)   
        print("end point:",x, y)
#        x_start=x_start+shift_vector[0]
#        y_start=y_start+shift_vector[1]
#        x_end=x_end+shift_vector[0]
#        y_end=y_end+shift_vector[1]
        ax.plot([x_start,y_start],[x_end,y_end],color,linewidth=line_width)
        ax.plot(xs[i],ys[i],color,linewidth=line_width)
    #    ax.plot(ns[i],rs[i],color,linewidth=line_width)
        return [x_start,y_start],[x_end,y_end],arc_length
    
def draw_circle(x_center,y_center,r):
    
    x_rs=[]
    y_rs=[]
    theta=0
    
    for theta in range(360):
        x_r=math.cos(math.radians(theta))*r+x_center
        y_r=math.sin(math.radians(theta))*r+y_center
        x_rs.append(x_r)
        y_rs.append(y_r)
        
    ax.plot(x_rs,y_rs,'b',linewidth=2)

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

test_number=10
xs = [[]]*test_number
ys = [[]]*test_number
rs = [[]]*test_number
ns = [[]]*test_number
fig = plt.figure()
fig.show()
ax = fig.add_subplot(1, 1, 1)
#ax.set_xlim(-10, 15)
#ax.set_ylim(-10, 15)
fig.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9)

j=1
step=0.01
factor_speed=1
factor_scale=1
#length=10
#curvstart=1/20.
#curvend=1/5.

x_test=[0,0,5,8]
y_test=[0,5,7,7]

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
#p_start,p_end,arc_length=draw_kurve(k_c[i-1],k_c[i],p_this,p_next,step,j,xs,ys,rs,ns,'r',1,ax)
p_start,p_end,arc_length=draw_kurve(1/10000,0.5,p_this,p_next,step,j,xs,ys,rs,ns,'r',1,ax)
print("p_start:",p_start)
print("p_end:",p_end)
print("arc_length:",arc_length)

#i+=1
#x_c[1],y_c[1],r_c[1],n_c[1]=find_circle(x_test,y_test,i)
#print(x_c[1],y_c[1],r_c[1])
#draw_circle(x_c[1],y_c[1],r_c[1])
#
#i+=1
#x_c[1],y_c[1],r_c[1],n_c[1]=find_circle(x_test,y_test,i)
#print(x_c[1],y_c[1],r_c[1])
#draw_circle(x_c[1],y_c[1],r_c[1])


