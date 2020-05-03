#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 20:14:34 2020

@author: cpprhtn
"""
#퍼셉트론 구현하기


#논리회로 AND
def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tmp= x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

AND(0,0)
AND(1,0)
AND(0,1)
AND(1,1)


#가중치 편향 도입
#b : 편향
#w : 가중치
import numpy as np
x = np.array([0,1])
w = np.array([0.5,0.5])
b = -0.7
w*x

np.sum(w*x)
np.sum(w*x) + b


#가중치와 편향을 도입한 AND gate
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
    
#NAND gate와 OR gate
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
    
    
    
#퍼셉트론의 한계
        #단층 퍼셉트론으로는 XOR gate를 표현할 수 없다.
        #단층 퍼셉트론으로는 비선형 영역을 분리할 수 없다.
        #가중치를 설정하는 작업은 사람이 수동으로 해야한다.
    
    
#XOR gate 구현하기 -2층 퍼셉트론
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y



'''
퍼셉트론은 입출력을 갖춘 알고리즘. 입력을 주면 정해진 규칙에 따른 값 출력.
퍼셉트론에서는 '가중치'와 '펀향'을 매개변수로 설정.
퍼셉트론으로 AND, OR 게이트 등의 논리 회로를 표현가능.
XOR 게이트는 단층 퍼셉트론으로는 표현할 수 없음.
2층 퍼셉트론을 이용하면 XOR 게이트를 표현가능.
단층 퍼셉트론은 직선형 영역만 표현할 수 있고, 다층 퍼셉트론은 비선형 영역도 표현할 수 있음.
다층 퍼셉트론은 (이론상) 컴퓨터를 표현할 수 있음.
'''



