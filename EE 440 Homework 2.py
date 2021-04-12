#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[26]:


img = cv2.imread('2_1.bmp')
cv2.imshow('Lena', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

datatype = img.dtype
img_max = np.amax(img)
print("The max is", img_max)
img_min = np.amin(img)
print("The min is", img_min)
print(img.shape)

count = np.zeros(256)

for i in range(0, 512, 1):
    for j in range(0, 512, 1):
        value = img[i, j]
        count[value] = count[value] + 1

plt.title('Histogram of Pixel Values')
plt.bar(np.arange(0,256,1), count)
plt.show()

pdfValues = count / (512 * 512)
plt.title('Probability Distribution Function of Pixel Values')
plt.bar(np.arange(0,256,1), pdfValues)
plt.show()

cdfValues = np.cumsum(pdfValues)
plt.title('Cumulative Distribution Function of Pixel Values')
plt.plot(np.arange(0,256,1), cdfValues)
plt.show()


# In[ ]:





# In[ ]:




