#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:21:44 2020

@author: cpprhtn
"""


'''
신경망 학습에서 훈련 데이터로부터 일부만 골라 학습을 수행할때, 이 일부를 미니배치 라고 한다.
예를 들어 60000개의 데이터중 100개를 무작위로 뽑아 그 100개만 사용하는 이러한 방법을 
미니배치 학습 이라고한다.
'''
#MINST 데이터를 불러옴
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)
    
print(x_train.shape)
print(t_train.shape)

#훈련데이터에서 무작위로 10장만 빼내기
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
np.random.choice(60000, 10)
#Out: array([ 4641,  2393, 30692, 44116, 52576,   657, 33029, 44298,  2569,
       24322])
    
#배치용 교차 엔트로피 오류 구현
'''
미니배치 같은 배치 데이터를 지원하는 교차 엔트로피 오차 함수
'''
#y가 신경망 t는 정답 레이블
#y가 1차원, 즉 데이터 하나당 교차 엔트로피 오차를 구하는 경우는 reshape 함수로 데이터의 형상을 바꿔준다
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        
        y = y.reshape(1, y.size)
        
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

'''
정답 레이블(t) 가 원-핫 인코딩이 아니라 2, 7 등의 숫자 레이블로 주어졌을 때의 교차 엔트로피오차
'''
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        
        y = y.reshape(1, y.size)
        
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
#이 구현에서는 원-핫 인코딩일 때 t가 0인 원소는 교차 엔트로피 오차도 0이므로 그 계산은 무시해도 된다는 뜻
#즉 정답에 해당하는 신경망의 출력만으로 교차 엔트로피 오차를 계산할 수 있다.
    

'''신경망을 학슬할 때 정확도를 지표로 삼아서는 안 된다.
정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이 되기 때문이다'''

    

