#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:02:41 2020

@author: cpprhtn
"""


'''
교차 엔트로피 오차 또한 손실 함수로 자주 사용한다
'''
#교차 엔트로피 오차
import numpy as np
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
cross_entropy_error(np.array(y), np.array(t))
#Out: 0.510825457099338

