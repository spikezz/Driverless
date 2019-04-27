#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:39:31 2019

@author: spikezz
"""

import numpy
a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
b = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
numpy.savetxt("foo.csv", a, delimiter=",")
numpy.savetxt("foo.csv", b, delimiter=",")