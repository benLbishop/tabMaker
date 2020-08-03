# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 21:07:43 2017

@author: bbishop
"""

import pyaudio
import wave
import numpy as np
from scipy import signal
import sys 
import struct
import matplotlib.pyplot as plt
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "writeTest2.wav"


f = sys.argv[1]
waveF = wave.open(f, 'r')
frames = list()

max_length = waveF.getnframes()      #total number of frames in wave file

for i in range(0, max_length):
    waveData = waveF.readframes(1)
    #frames.append(waveData)
    #frames.append(waveData)

    data = struct.unpack("hh", waveData)    # "hh" means unpack the data into two 16-bit signed ints
    n_data = (data[0], data[1])     #division to normalize data
    frames.append(n_data)


ch1, ch2 = zip(*frames)     #unpack the tuple into the separate channels

plt.plot(ch1)

#peakind = signal.find_peaks_cwt(ch1, np.arange(881,882))
#print "num peaks: {}".format(len(peakind))
#plt.plot(np.arange(len(ch1)), ch1, '-bo', markevery=peakind)
#plt.show()
#USING TIME DOMAIN (DOESN'T REALLY WORK)
"""
x1 = list()
x2 = list()
for frame in ch1:
    x1.append(2.0*frame)
    x1.append(frame)
for frame in ch2:
    x2.append(frame)
    x2.append(2.0*frame)
    
final = list()
for i in range(len(x1)):
    y = struct.pack("hh", x1[i], x2[i])
    final.append(y)
"""
#USING FOURIER TRANSFORM

c1 = np.fft.fft(ch1)
c2 = np.fft.fft(ch2)
#plt.plot(ch1)
expand_c1 = list()
expand_c2 = list()

for val in c1:
    expand_c1.append(0.5*val)
    expand_c1.append(0.5*val)
for val in c2:
    expand_c2.append(0.5*val)
    expand_c2.append(0.5*val)

x1 = np.fft.ifft(expand_c1)
x2 = np.fft.ifft(expand_c2)

for i in range(len(x1)):
    #remove imaginary component
    x1[i] = int(np.real(x1[i]))
    x2[i] = int(np.real(x2[i]))

plt.figure()
plt.plot(x1)
plt.show()    
final = list()
for i in range(len(x1)):
    y = struct.pack("hh", x1[i], x2[i])
    final.append(y)

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(waveF.getsampwidth())
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(final))
waveFile.close()