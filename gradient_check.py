#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:49:54 2020

@author: cpprhtn
"""


'''
기울기 검증하기
    two_layer_net(backpropagation)에서 오차역전파법으로 기울기를 구했다
    그러나 이 기울기가 맞는지 검증하기 위해서 수치 미분을 써본다
    
    수치 미분은 오차역전파법보다 속도가 훨씬 느리지만, 구현하기 쉽기때문에 에러가 잘 뜨지 않기때문이다
    
    구현과 사용은 오차역전파법을 쓰고, 이를 수치 미분으로 검증만 하면 된다
'''
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
