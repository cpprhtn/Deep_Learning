#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:08:55 2020

@author: cpprhtn
"""


#3층 신경망 구현하기

    #0층(입력층)에서 1층으로 신호전달
import numpy as np
X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1

#시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

Z1 = sigmoid(A1)


    #1층에서 2층으로 신호전달
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)


    #2층에서 3층(출력층)으로 신호전달
def identity_function(x):
    return x

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2 , W3) + B3
Y = identity_function(A3)



#구현 정리
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network

def forward(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y


network = init_network()
x = np.array([1.0,0.5])
y = forward(network ,x)
print(y) 
#[0.31682708 0.69627909]