#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:26:47 2020

@author: cpprhtn
"""


'''
최적화 방법
    Momentum(모멘텀) - 운동량
    
    아무런 힘을 받지 않았을때, 서서히 하강
    공이 그릇 바닥을 구르듯 움직임
'''
class Momentum:


    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]