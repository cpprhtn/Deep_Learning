#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 22:35:15 2020

@author: cpprhtn
"""


'''
사람이 생각한 특징 (SIFT, HOG 등) -> 기계학습(SVM, KNN 등) -> 결과

딥러닝을 종단간 기계학습 이라고도 함
'''


#손실함수
    #신경망 성능의 '나쁨'을 나타내는 지표
 

#평균 제곱 오차
    #가장 많이 쓰이는 손실함수
import numpy as np
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

def mean_squared_error(y, t):
    return (0.5 * (np.sum((y-t)**2)))

mean_squared_error(np.array(y), np.array(t))
#Output 0.09750000000000003

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mean_squared_error(np.array(y), np.array(t))
#Output 0.5975

#여기서 첫번째의 손실 값이 작으므로 오차가 작은걸 알 수 있다