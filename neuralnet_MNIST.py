#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:34:02 2020

@author: cpprhtn
"""


'''
MINST data set 을 이용하여 추론을 구현한느 신경망을 구현

입력층 뉴런 784개 (이미지 크기인 28*28)
출력층 뉴런 10개 (숫자 0에서 9까지 구별할 것이기 때문)

첫 번째 은닉층 : 뉴런 50개 (임의값)
두 번째 은닉층 : 누런 100개 (임의)
'''

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():    #sample_weight.pkl에 저장된 '학습된 가중치 매개변수'를 읽어옴
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):    #각 레이블의 확률을 넘파이로 반환
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:   #예측 적중시 count를 셈
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
#Accuracy:0.9352
 