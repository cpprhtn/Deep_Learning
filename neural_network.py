#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:15:58 2020

@author: cpprhtn
"""


#신경망
    #일반적으로 단순 퍼셉트론은 단층 네트워크에서 계단 함수(임계값을 경게로 출력이 바뀌는 함수)를 
    #활성화 함수로 사용한 모델을 가르키고, 다층 퍼셉트론은 신경망(여러 층으로 구성되고 
    #시그모이드 함수 등의 매끈한 활성화 함수를 사용하는 네트워크)을 말함.


#활성화 함수

#1 시그모이드 함수
#신경망에서 자주 이용하는 활성화 함수
h(x) = 1 / (1 + exp(-x))


#2 계단 함수 구현
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    #numpy배열은 받을 수 없음.
import numpy as np
def step_function(x):
    y = x > 0:
        return y.astype(np.int)
    #넘파이 배열에서 부등호 연산을 할 경우 bool형식으로 변환된 새로운 배열이 되는데
    #이를 다시 int형으로 바꿔주면 된다.


#3 계단 함수의 그래프
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0,dtype=np.int)

x = np.arange(-5.0,5.0,0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

#4 시그모이드 함수 구현
import numpy as np
import matplotlib.pylab as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()