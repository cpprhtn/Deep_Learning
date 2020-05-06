#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:11:50 2020

@author: cpprhtn
"""


#항등함수와 소프트맥스 함수

    #항등함수는 입력을 그대로 출력
    #즉 입출력이 똑같다

    
    #소프트맥스 함수는 분류에서 사용

a = np.array([0.3,2.9,4.0])

exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)
    

    #소프트맥스 ver.1
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

#소프트맥스 함수 구현 시 주의점
    #지수함수를 사용하는 소프트맥스에서 큰 값끼리 연산이 되어 오버플로 문제가 생길 수 있다.


    #소프트맥스 ver.2
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y





'''
기계학습의 문제 풀이는 학습과 추론의 두 단계를 거쳐 이뤄짐. 학습단계에서 모델을 학습하고,
추론 단계에서 앞서 학습한 모델로 미지의 데이터에 대해서 추론을 수행. 추론 단계에서는 출력층의
소프트맥스 함수를 생략하는 것이 일반적. 그러나 신경망을 학습시킬 때는 출력층에서 소프트맥스 함수를 사용
'''