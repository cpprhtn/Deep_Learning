#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:42:13 2020

@author: cpprhtn
"""


##비선형 함수
    #앞에서 공부했던 계단함수와 시그모이드 함수 모두 비선형 함수이다.
    #신경망에서는 활성화 함수로 주로 비선형 함수를 사용해야 한다.
    #왜냐하면 선형함수를 사용하면 신경망의 층을 깊게하는 의미가 없어지기 때문이다.


#ReLU(렐루)(Rectified Linear Unit) 함수
    #입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수
import numpy as np
def relu(x):
    return np.maximum(0,x)
