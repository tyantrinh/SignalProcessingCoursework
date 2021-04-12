#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Question 1

# In[4]:


img = cv2.imread('3_1.bmp')
cv2.imshow('3_1 image before negative', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for row in range(0, 512, 1):
    for col in range(0, 512, 1):
        img[row, col] = neg(img[row,col])

cv2.imshow('3_1 image after negative', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


def neg(i):
    first = 2 ^ 8
    second = first - i
    final = second - 1
    return final


# Question 2

# In[5]:


#part a
X = cv2.imread('3_2.bmp')
cv2.imshow('original', X)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(X.shape)

blue = X[:,:, 0]
cv2.imshow('blue image', blue)
cv2.waitKey(0)
cv2.destroyAllWindows()

green = X[:,:, 1]
cv2.imshow('green image', green)
cv2.waitKey(0)
cv2.destroyAllWindows()

red = X[:,:, 2]
cv2.imshow('red image', red)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


#part b
im = cv2.imread('3_2.bmp')
imConvert = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

cv2.imshow('original', imConvert)
cv2.waitKey(0)
cv2.destroyAllWindows()


H = imConvert[:,:, 0]
cv2.imshow('H image', H)
cv2.waitKey(0)
cv2.destroyAllWindows()

S = imConvert[:,:, 1]
cv2.imshow('S image', S)
cv2.waitKey(0)
cv2.destroyAllWindows()

V = imConvert[:,:, 2]
cv2.imshow('V image', V)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




