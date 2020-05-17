#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:15:57 2020

@author: cpprhtn
"""

'''
기울기(gradient) : 모든 변수의 편미분을 벡터로 정리한 것

기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향
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

def function_2(x):
    return x[0]**2 + x[1]**2
#f2(x)함수에서 (3,4)의 기울기 계산
numerical_gradient(function_2, np.array([3.0,4.0]))
#Out: array([6., 8.])