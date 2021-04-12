#!/usr/bin/env python
# coding: utf-8

# # Lab 3 Report
# 
# Cedric Kong & Tyan Trinh

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile as wav
from scipy import signal as sig
import simpleaudio as sa
import decimal


# ## Summary
# 
# This lab includes wokring with unit steps and impulse responses. Also includes playing and plotting wav fileds with different delays and colvoluting and summing resulting signals to recover the original audio file. 

# ## Assignment 1 -- Convolving Signals

# In[11]:


# Part A
fs = 1000
t = np.arange(0, 4.1, 1/fs)
def u(t): 
    return 1.0*(t>0)
def delta(t,fs):
    return fs*np.concatenate([[0],np.diff(u(t))])
x = u(t-1)-u(t-3)
h1 = delta(t-1, fs)
h2 = u(t)-u(t-1)
h3 = u(t)-2*u(t-0.5)+u(t-1)

# Part B
fig1 = plt.figure(1)
fig1.subplots_adjust(hspace = 1.5, wspace = 1.5)

plt.subplot(3, 1, 1)
plt.plot(t, h1, color = 'r')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('h1 vs Time')
plt.xlim(0, 4)
plt.ylim(-2.5, 2.5)

plt.subplot(3, 1, 2)
plt.plot(t, h2, color = 'b')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('h2 vs Time')
plt.xlim(0, 4)
plt.ylim(-2.5, 2.5)

plt.subplot(3, 1, 3)
plt.plot(t, h3, color = 'g')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('h3 vs Time')
plt.xlim(0, 4)
plt.ylim(-2.5, 2.5)

# Part C
y1 = np.convolve(x, h1)/fs
y2 = np.convolve(x, h2)/fs 
y3 = np.convolve(x, h3)/fs 

# Part D
ty = np.arange(0, 8.199, 1/fs)

x = np.concatenate([x, np.zeros(len(x) - 1)])

fig2 = plt.figure(2)
fig2.subplots_adjust(hspace = 1.5, wspace = 1.5)

plt.subplot(4, 1, 1)
plt.plot(ty, x, color = 'r')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('x vs Time')
plt.xlim(0, 8)
plt.ylim(-2, 2)

plt.subplot(4, 1, 2)
plt.plot(ty, y1, color = 'b')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('y1 vs Time')
plt.xlim(0, 8)
plt.ylim(-2, 2)

plt.subplot(4, 1, 3)
plt.plot(ty, y2, color = 'g')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('y2 vs Time')
plt.xlim(0, 8)
plt.ylim(-2, 2)

plt.subplot(4, 1, 4)
plt.plot(ty, y3, color = 'k')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('y3 vs Time')
plt.xlim(0, 8)
plt.ylim(-2, 2)


# ### Assignment 1 Discussion
# 
# When you use a sampling rate of 1000 instead of 10, the changes in the graph look much more instantaneous. So, when you use a sampling rate of 1000, it actually looks like a unit step function or a delta function.

# ## Assignment 2 -- Revisiting Time Delay Transformation 

# In[16]:


# Assignment 2

# Part A
fs, x = wav.read('train32.wav')
t_x = np.arange(0, len(x), 1) * (1/fs)

# Part B
t_h = np.arange(0, 2.1, 1/fs)
impulse_n = int(1 * fs)
hd = np.zeros(len(t_h))
hd[impulse_n] = fs

# Part C
y = np.convolve(x, hd) / fs
t_y = np.arange(0, len(y), 1) * (1/fs)

# Part D
fig3 = plt.figure(3)
fig3.subplots_adjust(hspace = 1.5, wspace = 1.5)

plt.subplot(3, 1, 1)
plt.plot(t_x, x, color = 'r')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('x vs Time')
plt.xlim(0, 4)

plt.subplot(3, 1, 2)
plt.plot(t_h, hd, color = 'b')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('hd vs Time')
plt.xlim(0, 4)

plt.subplot(3, 1, 3)
plt.plot(t_y, y, color = 'g')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('y vs Time')
plt.xlim(0, 4)


# ### Assignment 2 Discussion
# 
# If you modify C by not scaling according to the sampling time, the shift stays the same, but the amplitude of the graph exponentially increases compared to the original y(t). The output signal would be much louder and be static compared to the original y(t).

# ## Assignment 3 -- Audio File Realignment

# In[23]:


# Assignment 3

# Part A
fs, x1 = wav.read('s2_1.wav')
x1 = x1[:,0] # Channel 0

fs, x2 = wav.read('s2_2.wav')
x2 = x2[:,0]

fs, x3 = wav.read('s2_3.wav')
x3 = x3[:,0]

# Part B
# Order is s2_2, s2_1, s2_3. All clips are 1 sec long. s2_1 should be delayed
# 1 sec, and s2_3 should be delayed 2 sec.
t = np.arange(0, 3.1, 1/fs)

impulse_n1 = int(1 * fs)
h1 = np.zeros(len(t))
h1[impulse_n1] = fs

h2 = np.zeros(len(t))
h2[0] = fs

impulse_n3 = int(2 * fs)
h3 = np.zeros(len(t))
h3[impulse_n3] = fs
print(h3)

# Part C
y1 = np.convolve(x1, h1)/fs
y2 = np.convolve(x2, h2)/fs 
y3 = np.convolve(x3, h3)/fs 
print (len(y1)) # "I am"
print (len(y2)) # "Luke"
print (len(y3)) # "Your father"

# Part D
y1 = np.concatenate([y1, np.zeros(len(y3) - len(y1))])
y2 = np.concatenate([y2, np.zeros(len(y3) - len(y2))])
y4 = y1 + y2 + y3

outfile1 = 'LUKE.wav'
wav.write(outfile1, fs, y4.astype('int16'))
wave_obj = sa.WaveObject.from_wave_file(outfile1)
play_obj = wave_obj.play()
play_obj.wait_done()


# ### Assignment 3 Discussion
# 
# If you do x2(t) instead, assuming x2 is the shortest length, you would lose parts of the famous quotes because they are being shortened to match the length of x2. If x2 is the longest length instead, you would have gaps. Star Wars- Darth Vader, "Luke, I am your father!" Iconic.

# In[ ]:




