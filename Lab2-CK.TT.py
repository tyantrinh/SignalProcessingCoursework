#!/usr/bin/env python
# coding: utf-8

# Lab 2 Report
# Cedric Kong & Tyan Trinh

# In[19]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile as wav
from scipy import signal as sig
import simpleaudio as sa
import decimal


# Assignment 1: Time Scaling Audio Signal
# Use timescale to change how how long the audiosample is played. Changes how it sounds.

# In[7]:


# Assignment 1 - Time Scaling Audio Singal
# Part A
def timescale(x,fs,a):
    n,d = decimal.Decimal(a).as_integer_ratio()
    y= sig.resample_poly(x,d,n)
    t=np.arange(0,len(y),1)*(1/fs)
    return y,t

# Part B
wav_obj = sa.WaveObject.from_wave_file('train32.wav')
fs = wav_obj.sample_rate
y = wav_obj.num_channels

fs, data = wav.read('train32.wav')

t_y = np.arange(0, 50313) / fs

# Part C
w, t_w = timescale(data,fs,2)
v, t_v = timescale(data,fs,0.5)

outfile = 'w.wav'
wav.write(outfile, fs, w.astype('int16'))
outfile = 'v.wav'
wav.write(outfile, fs, v.astype('int16'))

# wav_objw = sa.WaveObject.from_wave_file('w.wav')
# play_obj = wav_objw.play()
# play_obj.wait_done()

# wav_objv = sa.WaveObject.from_wave_file('v.wav')
# play_obj = wav_objv.play()
# play_obj.wait_done()

# play_obj = wav_obj.play()
# play_obj.wait_done()

# Part D
fig3 = plt.figure(3)
fig3.subplots_adjust(hspace = 0.1, wspace = 0.5)

plt.subplot(3,1,1)
plt.plot(t_w, w, color='k')
plt.title('y(2t)')
plt.xlabel('time (s)')
plt.ylim(-30000, 30000)
plt.xlim(0,4)


plt.subplot(3,1,2)
plt.plot(t_v, v, color = 'k')
plt.title('y(0.5t)')
plt.xlabel('time (s)')
plt.ylim(-30000, 30000)
plt.xlim(0,4)

plt.subplot(3,1,3)
plt.plot(t_y, data, color = 'k')
plt.title('y(t)')
plt.xlabel('time (s)')
plt.ylim(-30000, 30000)
plt.xlim(0,4)


# Assignment 2 - Time Shift Operation

# In[39]:


# Part A

def timeshift(x, fs, t0):
    x = x.astype('int16')
    
    n0 = np.zeros(t0 * fs)
    if t0 < 0.0:
        y = np.concatenate([n0,x])
    else:
        x = x[n0: len(x)]
        y = np.concatenate([x,n0])
        
    t = np.arange(0, len(y), 1) * (1/fs)
    return y, t        


# In[42]:


# Part B

y1, t1 = timeshift(y,fs,0.5)


# In[25]:


#Assignment 3
#Part A

S,W = wav.read('s1.wav')
St = np.arange(0,len(W),1)*(1/S)

play_obj = sa.play_buffer(W,1,2,S)
play_obj.wait_done()

ya,ta = timescale(W,S,2)

ya= ya.astype('int16')
play_obj = sa.play_buffer(ya,1,2,S)

play_obj.wait_done()

yb,tb = timescale(W,S,0.5)
yb = yb.astype('int16')

print('Terminator')

yb, tb = timeshift(yb,S,-0.5)
yb = yb.astype('int16')

play_obj = sa.play_buffer(yb,1,2,S)


# In[31]:


#Part C
fig4 = plt.figure(4)
fig4.subplots_adjust(hspace = 1, wspace = .1)

plt.subplot(2,1,1)
plt.plot(St,W, label = 'S(t)')
plt.ylim(-5000, 5000)
plt.xlim(0,4)
plt.title('what the fuck')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2,1,2)
plt.plot(tb, yb, label = 'yb(t)')
plt.ylim(-5000, 5000)
plt.xlim(0,4)
plt.title('Unaltered')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




