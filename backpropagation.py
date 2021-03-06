#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:57:31 2020

@author: cpprhtn
"""


'''
오차역전파법 : 가중치 매개변수의 기울기를 효율적으로 계산


계산그래프 : 계산 과정을 그래프로 나타냄
    노드와 에지로 표현
    
    계산그래프의 흐름
        1. 계산그래프 구성
        2. 그래프에서 계산을 왼쪽에서 오른쪽으로 진행 (순전파)
        ~2 (역전파)
        
    계산그래프의 특징
        국소전 계산을 전파함으로써 최종 결과를 도출
        
        국소적 미분을 전달하는 원리는 연쇄법칙을 따름
        
            연쇄법칙 : 합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타냄
'''
