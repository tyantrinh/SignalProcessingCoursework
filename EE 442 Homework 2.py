#!/usr/bin/env python
# coding: utf-8

# In[117]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack


# ## Problem 1

# In[118]:


def u(n):
    return 1 * (n > 0)


# In[119]:


n = np.arange(-5, 5)


x1 = ((0.5)**np.abs(n)) * (u(n+10) - u(n-10))


x1 = np.concatenate((x1, np.zeros(900)))
x1fft = fftpack.fft(x1)
x1final = fftpack.fftshift(x1fft)

n2 = np.arange(0, 300)
x2 = (n2)*(0.9)**n2
x2fft = fftpack.fft(x2)
x2final = fftpack.fftshift(x2fft)

f1 = np.linspace(-np.pi, np.pi, len(x1final))
f2 = np.linspace(-np.pi, np.pi, len(x2final))


# In[120]:


plt.title("Magnitude of X1")
plt.plot(f1, np.abs(x1final))
plt.xlabel('Frequency')
plt.ylabel('Magnitude')


# In[121]:


plt.title("Phase of X1")
plt.plot(f1, np.angle(x1final))
plt.xlabel('Frequency')
plt.ylabel('Magnitude')


# In[122]:


plt.title("Magnitude of X2")
plt.plot(f2, np.abs(x2final))
plt.xlabel('Frequency')
plt.ylabel('Magnitude')


# In[123]:


plt.title('Phase of X2')
plt.plot(f2, np.angle(x2final))
plt.xlabel('Frequency')
plt.ylabel('Magnitude')


# ## Problem 3

# In[124]:


n = np.arange(-5, 5)
x = 2 * np.exp(-.9 * np.abs(n))

x = np.concatenate((x, np.zeros(900)))

xfft = fftpack.fft(x)
xfftshift = fftpack.fftshift(xfft)

f2 = np.linspace(-np.pi, np.pi, len(xfftshift))
plt.plot(f2, np.abs(xfftshift))
plt.title("DFT of Xn")
plt.ylabel("Magnitude")
plt.xlabel("Frequency")


# In[ ]:




