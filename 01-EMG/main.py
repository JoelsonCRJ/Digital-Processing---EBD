import scipy.fftpack
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO 

import sys
sys.path.insert(1, r'./../functions')  # add to pythonpath
from detect_peaks import detect_peaks


filepath = "coleta_valnete.txt"  
a=np.loadtxt(filepath)	
data = np.array(a[:,0])
time=np.array(a[:,1])


#plt.subplot(2, 1, 1)
#plt.plot(time, data, )
#plt.title('aquisição dos dados')
#plt.ylabel('sinal quantizado')

Fs=1000.0
T = 1/Fs

b, a = signal.butter(5, [10/500, 450/500], 'bandpass')
Y_butter = lfilter(b, a,data)
#plt.subplot(2, 1, 2)
#plt.plot(time, Y_butter)
#plt.xlabel('time (s)')
#plt.ylabel('amplitude')
#plt.show()

N=len(Y_butter)
t=N*T

Y=scipy.fft(Y_butter)
P2=abs(Y/N)
a=int (N/2+1)
b=int (N/2)
P1=P2[2:a]

P1[2:(N-1)]=2*P1[2:(N-1)]
f = (Fs/N)*np.arange(0,a,1);
#plt.plot(2)
#plt.plot(f[2:a],P1)
Inv = scipy.ifft(Y)
ind = detect_peaks(Inv,mph=0, mpd=500,show=True)

#plt.subplot(2, 1, 2)
#plt.plot(time, Inv)
#plt.xlabel('time (s)')
#plt.ylabel('amplitude')
#
#plt.show()

#L = length(Y_butter)
#t = length(Y_butter)*T

#YBT_FFT=scipy.fft(Y_butter)
#P2_YBT = abs(Y_butter/L);
#a_YBT=int ((N/2)+1)
#P1_YBT = P2[1:a_YBT]
#P1_YBT[2:L-1] = 2*P1_YBT[2:L-1];
#f_YBT = (Fs/L)*np.arange(0,(L/2),1);



#N=len(data)

#yf = scipy.fftpack.fft(data)
#P2 = abs(yf/N);
#a=int ((N/2)+1)
#print(a)
#P1 = P2[1:a]

#P1[2:N-1] = 2*P1[2:N-1];
#print(len(P1))
#f = (Fs/N)*np.arange(0,(N/2),1);
#print(len(f))
#plt.plot(f,P1)
#plt.grid()
#plt.show()


