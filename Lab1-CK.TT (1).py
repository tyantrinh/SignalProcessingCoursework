#!/usr/bin/env python
# coding: utf-8

# # Lab 1 Report
# 
# Cedric Kong & Tyan Trinh 
# 
# (This should be a markup cell, which means that when you run it you just get formated text.)

# In[2]:


# We'll refer to this as the "import cell." Every module you import should be imported here.
get_ipython().run_line_magic('matplotlib', 'notebook')
# example: importing numpy
import numpy as np
import simpleaudio as sa
import matplotlib
import matplotlib.pyplot as plt

# import whatever other modules you use in this lab

# The commands you use to make your modules usable also go in this cell.


# ## Summary
# 
# This lab includes working with arrays, extracting and manipulating wav files, playing wav files with different frequencies, as well as plotting wav files.

# ## Assignment 1 -- Creating and Manipulating Arrays (Title of Assignment 1)

# In[25]:


# Assignment 1 - Creating and Manipulating Arrays (Title of Assignment 1)

# Here you'd put in your code for Assignment 1, organized by parts if necessary
# For example:

# Part A

y1 = np.array([4, 6, 2])

# Part B
ca = y1[1]
cb = y1[1:3]
d = len(y1)
print("ca = ", ca, "cb = ", cb, "d = ", d)

# Part C
x1 = 2*np.ones(5)
x2 = np.arange(-2,3)
print(x1, x2)

# Part D
arrp = x1 + x2
arrc = np.concatenate([x1,x2])
print(arrp, arrc)


# ###  Discussion
# 
# Concatenation appends one vector to the end of another. The concatenated vector will have the length of the two vector. Comparatively, adding vectors will add element by element, resulting in vector addition with the same vector length.
# 
# One constraint when adding two vectors must be the same length. This does not apply to concatenation.
# A contraint that applies to both is that the dimensions of the data must match.
# 

# ## Assignment 2 -- Amplitude Operations on Signals (title of assignment 2)

# In[94]:


# Assignment 2 - Title of Assignment 2

# Here you'd put in your code for Assignment 2

# Part A
fs=2 # example of defining the sampling frequency
t = np.arange(0, 3.5, 1/fs)
x = 0.5 * t
y = t * t
print('fs = ', fs) # Print out fs
print('t = ', t)
print(x, y)

# Part B
z = x - 2*y
print(z)

# Part C
w1 = z[4]
print(w1)

# Part D
w2 = z[0:4]
print(w2)


# ###  Discussion
# 
# Sampling frequency of 2 gives us 2 data points per second. Sampling frequency of 1 gives us only 1 data point per second.

# ## Assignment 3 -- Working with Sound Files

# In[3]:


# Assignment 3

# Part A
wav_obj = sa.WaveObject.from_wave_file('train32.wav')
fs1 = wav_obj.sample_rate
y1 = wav_obj.num_channels
print('Sampling frequency =', fs1)
print('Number of channels =', y1)

wav_obj2 = sa.WaveObject.from_wave_file('tuba11.wav')
fs2 = wav_obj2.sample_rate
y2 = wav_obj.num_channels
print('Sampling frequency =', fs2)
print('Number of channels =', y2)

from scipy.io import wavfile as wav
fs1, data1 = wav.read('train32.wav')
print('Train whistle has: sample rate', fs1, ', # of samples', len(data1), ', type', data1.dtype)
fs2, data2 = wav.read('tuba11.wav')
len2, ch2 = data2.shape
print('Tuba has: sample rate', fs2, ', # f samples', len2, ', # of channels', ch2, ', type', data2.dtype)

# Part B
play_obj = wav_obj.play()
play_obj.wait_done()

play_obj = wav_obj2.play()
play_obj.wait_done()

#wav_obj.sample_rate = fs2
play_obj = wav_obj.play()
play_obj.wait_done()

wav_obj2.sample_rate = fs1
play_obj = wav_obj2.play()
play_obj.wait_done()

# Part C
data3 = data2[:,1]
y3 = data3[0:50313]
y4_data = np.concatenate([data1, y3])
outfile = 'y4.wav'
wav.write(outfile,fs1,y4_data.astype('int16'))
print(len(data1), len(y3), len(y4))

wav_obj3 = sa.WaveObject.from_wave_file('y4.wav')
play_obj = wav_obj3.play()
play_obj.wait_done()

# Part D
pause = np.zeros(int(4))
y5_data = np.concatenate([data1, pause, data2[:,1], pause, data2[:,0]])
outfile = 'y5.wav'
wav.write(outfile, fs1, y5_data.astype('int16'))

wav_obj4 = sa.WaveObject.from_wave_file('y5.wav')
play_obj = wav_obj4.play()
play_obj.wait_done()


# In[ ]:





# ### Discussion
# 
# Increasing sampling frequency makes the pitch higher, while decreasing the sampling frequency makes the pitch lower. For example, the tuba sounded like a trumpet when played with a higher sampling frequency. The length of the audio is also affected by the sampling frequency. Sampling a file with a faster than expected frequency will result in a shorter output.
# 
# As explained earlier, the dimensions must match in order to concatenate. In sections C and D, we only worked with one tuba channel so that it matched the 1-channel train file. 
# 

# ## Assignment 4 -- Plotting Comparisons

# In[101]:


# Assignment 4

# Part A
t = np.arange(-2, 5)
w = abs(t)
x = 2-t
y = -0.5*t*t

# Part B
plt.plot(t, w, '-')
plt.plot(t, x, '--')
plt.plot(t, y, '.')

plt.title('x(t), y(t), w(t) vs time graph')
plt.xlabel('time')


# ### Discussion
# 
# the plt.plot() functions are capable of changing the line pattern. '-', '--' or '.' are examples of plot styles that can be used to help color blind readers. 

# ## Assignment 5 -- Plotting Sound Files using Subplots
# 

# In[104]:


# Assignment 5 -- Plotting Sound Files using Subplots

# Part A
from scipy.io import wavfile as wav
fs1, data1 = wav.read('train32.wav')
fs2, data2 = wav.read('tuba11.wav')

data3 = data2[:,1]
y3 = data3[0:50313]

timeArray = np.arange(0, 50313)
t1 = timeArray / fs1 * 1000
t2 = timeArray / fs2 * 1000

# Part B
fig2 = plt.figure(2)
fig2.subplots_adjust(hspace = 0.1, wspace = 0.5)

plt.subplot(2, 2, 1)
plt.plot(t1, data1, color='k')
plt.title('Data1 vs Time')
plt.xlabel('time (ms)')
plt.ylim(-35000, 35000)
plt.xlim(1000,1100)


plt.subplot(2,2,2)
plt.plot(t2, y3, color = 'k')
plt.title('Data2 vs Time')
plt.xlabel('time (ms)')
plt.ylim(-35000, 35000)
plt.xlim(1000,1100)


# ### Discussion
# 
# Differences:
#  - Sampling frequencies
#  - Magnitudes
#  - Time elapsed
# 
# The train graph (data1) has much more data points because of the almost 3x greater sampling frequency compared to the tuba graph (data2).

# In[ ]:




