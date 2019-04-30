#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:45:44 2019

@author: spikezz
"""

class Sample:
    def __enter__(self):
        print ("in __enter__")
        return "Foo"
    def __exit__(self, exc_type, exc_val, exc_tb):
        print ("in __exit__")
def get_sample():
    
    return Sample()

with get_sample() as sample:
    
    print ("Sample: ", sample)