#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:44:51 2020

@author: cpprhtn
"""



#편미분

#f(x_0, x_1) = (x_0)^2 + (x_1)^2 구현
def function_2(x):
    return x[0]**2 + x[1]**2
    #retrun np.sum(x**2)

#인수 x는 넘파이 배열이라고 가정
'''
편미분 : 변수가 여럿인 함수에 대한 미분
어느 변수에 대한 미분이냐를 구별할 필요가 있다.
'''

#문제 1 : x_0 = 3, x_1 = 4 일때, x_0에 대한 편미분을 구하라.

#수치 미분
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

numerical_diff(function_tmp1, 3.0)
#Out: 6.00000000000378


#문제 2 : x_0 = 3, x_1 = 4 일때, x_1에 대한 편미분을 구하라.
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

numerical_diff(function_tmp2, 4.0)
#Out: 7.999999999999119