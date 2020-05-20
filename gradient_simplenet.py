#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:54:27 2020

@author: cpprhtn
"""


'''
신경망에서의 기울기는 가중치 매개변수에 대한 손실 함수의 기울기
'''

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

'''
위 구현에서는 새로운 함수를 정의하는데 def ~ 문법을 썻지만
간단한 함수라면 람다 기법을 쓰는것이 더 편하다.
'''

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)
#[[ 0.11646816  0.24796626 -0.36443442]
# [ 0.17470224  0.37194939 -0.54665164]]


