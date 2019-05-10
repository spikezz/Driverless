#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:49:20 2019

@author: spikezz
"""

import csv

# 使用数字和字符串的数字都可以
datas = [['name', ['age','age_attachment']],
         ['Bob', [14,3]],
         ['Tom', [23,5]],
        ['Jerry', ['18','7']]]

with open('example.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in datas:
        writer.writerow(row)
        
    # 还可以写入多行
    writer.writerows(datas)