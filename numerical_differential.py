#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:59:47 2020

@author: cpprhtn
"""


'''
미분 구현
'''
#수치미분
    #나쁜 구현 예
def numerical_diff(f, x):
    h = 10e-50
    retrun (f(x + h) - f(x)) / h
    

'''
위 함수의 문제점

반올림 문제 : 매우 값이 작을경우 반올림이 적용되어 오차가 발생

함수 f의 차분 : 실제 접선과 위에서 구현한 기울기가 일치하지 않음
'''
    #개선된 수치 미분
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)



#0.01x^2 + 0.1x 함수 구현과 시각화
def function_1(x):
    return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()

#x = 5, x = 10에서의 미분
numerical_diff(function_1, 5)
#Out: 0.1999999999990898

numerical_diff(function_1, 10)
#Out: 0.2999999999986347


#미분값 시각화
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
