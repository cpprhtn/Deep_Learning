#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:09:28 2020

@author: cpprhtn
"""


'''
신경망
훈련데이터를 이용해 가중치 매개변수 학습
학습한 매개변수를 사용하여 입력데이터 분류(추론단계)
'''


#손글씨 숫자 인식
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)

#flatten=True로 설정해 읽어 들인 이미지는 1차원 넘파이 배열로 저장
#reshape()메서드에 원하는 형상을 인수로 지정하면 넘파이 배열의 형상을 바꿀 수 있음