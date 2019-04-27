#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:52:35 2019

@author: spikezz
"""

import numpy as np
# 实现插值的模块
from scipy import interpolate
# 画图的模块
import matplotlib.pyplot as plt
# 生成随机数的模块
import random

#from scipy.interpolate import BSpline

#def B(x, k, i, t):
#    if k == 0:
#        return 1.0 if t[i] <= x < t[i+1] else 0.0
#    if t[i+k] == t[i]:
#        c1 = 0.0
#    else:
#        c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
#    if t[i+k+1] == t[i+1]:
#        c2 = 0.0
#    else:
#        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
#    return c1 + c2
#
#def bspline(x, t, c, k):
#    n = len(t) - k - 1
#    assert (n >= k+1) and (len(c) >= n)
#    return sum(c[i] * B(x, k, i, t) for i in range(n))
#
#k = 2
#t = [0, 1, 2, 3, 4, 5, 6]
#c = [-1, 2, 0, -1]
#spl = BSpline(t, c, k)
#spl(2.5)
#
#fig, ax = plt.subplots()
#xx = np.linspace(1.5, 4.5, 50)
#ax.plot(xx, [bspline(x, t, c ,k) for x in xx], 'r-', lw=3, label='naive')
#ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
#ax.grid(True)
#ax.legend(loc='best')
#plt.show()
#

# random.randint(0, 10) 生成0-10范围内的一个整型数
# y是一个数组里面有10个随机数，表示y轴的值
#y = np.array([random.randint(0, 10) for _ in range(10)])
## x是一个数组，表示x轴的值
#x = np.array([num for num in range(10)])
y = np.array([0,3,3,7,7])
x = np.array([0,4,9,12,7])

theta=np.array([0,1,2,3,4])

# 插值法之后的x轴值，表示从0到9间距为0.5的18个数
thetanew=np.arange(0, 4, 0.01)

funcx = interpolate.interp1d(theta, x, kind='quadratic')
funcy = interpolate.interp1d(theta, y, kind='quadratic')

xnew = funcx(thetanew)
#ynew = func(thetanew)
#xnew = np.arange(0, 12, 0.01)

"""
kind方法：
nearest、zero、slinear、quadratic、cubic
实现函数func
"""
#func = interpolate.interp1d(x, y, kind='quadratic')
# 利用xnew和func函数生成ynew，xnew的数量等于ynew数量
ynew = funcy(thetanew)

# 画图部分
# 原图
fig = plt.figure()
fig.show()
ax = fig.add_subplot(1,1,1)
fig.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9)
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
plt.plot(x, y, 'ro-')
# 拟合之后的平滑曲线图
#plt.plot(thetanew, xnew)
plt.plot(xnew, ynew)
plt.show()
