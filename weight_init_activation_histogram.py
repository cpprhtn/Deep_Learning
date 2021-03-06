#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:48:12 2020

@author: cpprhtn
"""


'''
은닉층의 활성화값 분포
'''
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)
    
input_data = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 초깃값
        #정규분포
    #w = np.random.randn(node_num, node_num) * 1
    w = np.random.randn(node_num, node_num) * 0.01
        #사비에르 초깃값
    #w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
        #He 초깃값
    #w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)


    a = np.dot(x, w)


    # 활성화 함수
    #z = sigmoid(a)
    z = ReLU(a)
    #z = tanh(a)

    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()


'''
표준편차를 0.01로 한 정규분포는 가운데로 치우침
표준편차를 1로 한 정규분포는 끝으로 치우침
사비에르 초깃값을 사용할 경우 각 층에 흐르는 데이터의 양이 적당
'''


'''
활성화 함수로 ReLU를 사용할 때는 He 초깃값을,
sigmoid나 tanh 등의 S자 곡선일 때는 Xavier 초깃값을 쓰는것이 낫다
'''