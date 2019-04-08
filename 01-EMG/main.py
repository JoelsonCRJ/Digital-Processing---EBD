import scipy.fftpack
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(1, r'./../functions')  # add to pythonpath
from detect_peaks import detect_peaks
#------------------------------------------------------------------------

filepath = "coleta_valnete.txt"  
a=np.loadtxt(filepath)	
data = np.array(a[:,0])
time=np.array(a[:,1])
Fs=1000.0
T = 1/Fs

b, a = signal.butter(5, [10/500, 450/500], 'bandpass')
Y_butter = lfilter(b, a,data)

N=len(Y_butter)
t=N*T

Y=scipy.fftpack.fft(Y_butter)
# P2=abs(Y/N)
# a=int ((N/2)+1)
# P1=P2[2:a]
# P1[2:N-1] = 2*P1[2:N-1]
# f = (Fs/N)*np.arange(0,(N/2),1)
# plt.plot(f[0:len(f)-1],P1)
# plt.show()

Inv = scipy.ifft(Y)
ind = detect_peaks(Inv,mph=0, mpd=(0.8/T),show=True)







