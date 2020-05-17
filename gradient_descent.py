#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:27:56 2020

@author: cpprhtn
"""


'''
경사법(경사 하강법) : 최적의 매개변수를 학습시 찾아야 하는데 이런 상황에서 기울기를 잘 이용해
                 함수의 최솟값을 찾으려는 것
                 
                 
                 현 위치에서 기울어진 방향으로 일정 거리만큼 이동
                 이동한 곳에서도 마찬가지로 기울기를 구하고, 또 그 기울어진 방향으로 나아가기를 반복
                 이렇게 해서 함수값을 점차 줄이는 것
'''
import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad



#경사 하강법 구현
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


#문제 : 경사법으로 f(x_0, x_1) = (x_0)^2 + (x_1)^2의 최솟값을 구하라
def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0,4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
#Out: array([-1.25592487e-19,  1.66263303e-19]) (0,0)에 가깝다.