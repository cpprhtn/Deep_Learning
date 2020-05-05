#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:29:46 2020

@author: cpprhtn
"""


##다차원 배열

#1차원 배열
import numpy as np
A = np.array([1,2,3,4])
print(A)
np.ndim(A)  #배열의 차원수 확인
A.shape()   #배열의 형상 확인(튜플형식)
A.shape[0]


#2차원 배열
B = np.array([[1,2],[3,4],[5,6]])
print(B)
np.ndim(B)
B.shape


#행렬의 곱
    #대응하는 차원의 원소수가 같아야 한다
    #ex) 6,10 * 10,3 -> 6,3   

    #2,2 * 2,2 행렬
A = np.array([[1,2],[3,4]])
A.shape
B = np.array([[5,6],[7,8]])
B.shape

np.dot(A, B)

    #2,3 * 3,2 행렬
A = np.array([[1,2,3],[4,5,6]])
A.shape
B = np.array([[1,2],[3,4],[5,6]])
B.shape

np.dot(A, B)



#신경망에서의 행렬의 곱
X = np.array([1,2])
X.shape
W = np.array([[1,3,5],[2,4,6]])
print(W)
W.shape
Y = np.dot(X,W)
print(Y)