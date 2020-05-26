#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:05:37 2020

@author: cpprhtn
"""


'''
단순한 계층 구현
    신경망을 구현하는 계층 각각을 하나의 클래스로 구현
'''

#곱셈 계층
class MulLayer:
    def __init__(self): #인스턴스 변수 초기화
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout): #상류에서 넘어온 미분(dout)에 순전파 때의 값을 바꿔 곱한후 하류로 보냄
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy
    
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward_순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward_역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price) #220.00000000000003
print(dapple, dapple_num, dtax) #2.2 110.00000000000001 200
