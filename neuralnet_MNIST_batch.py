#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:16:10 2020

@author: cpprhtn
"""


'''
데이터를 특정 범위로 변환하는 처리를 정규화라고 하고, 
신경망의 입력 데이터에 특정 변환을 가하는 것을 전처리라 함

TMI
현업에서도 전처리를 통해 식별 능력을 개선하고 학습 솓도를 높이는 등의 사례가 많이 제시되고 있음
ex) 
1. 데이터 전체 평균과 표준편차를 이용하여 데이터들이 0을 중심으로 분포하도록 이동하거나 
       데이터의 확산 범위를 제한하는 정규화
2. 전체 데이터를 균일하게 분포시키는 데이터 백색화

'''

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

'''
배치 : 하나로 묶은 입력 데이터
배치 처리의 장
1. 수치 계산 라이브러리 대부분이 큰 배열을 효율적으로 처리할 수 있도록 최적화되어 있기 때문
2. 느린 I/O를 통해 데이터를 읽는 횟수가 줄어, CPU GPU로 순수 계산을 수행하는 비율 증가
'''


