import scipy.fftpack
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import sys
sys.path.insert(1, r'./../functions')  # add to pythonpath
from detect_peaks import detect_peaks
#------------------------------------------------------------------------

def remove_peaks(peaks_index_array,array,window_size):
    for i in range (0,len(peaks_index_array)):
        if (peaks_index_array[i]<window_size):
            array[0:peaks_index_array[i]+8*window_size]=0+0j
        elif((len(array)-i) < window_size):
            array[peaks_index_array[i] :(peaks_index_array[i]-2*window_size)]=0+0j
        else:
            array[(peaks_index_array[i]-50):(peaks_index_array[i]+50)] = 0+0j
    return array
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
# plt.plot(f[0:len(f)-1],P1)-1
# plt.show()

Inv = scipy.ifft(Y)
ind = detect_peaks(Inv,mph=0, mpd=(0.8/T),show=True)

Inv=remove_peaks(ind,Inv,50)

# f=interp1d(time[0:400].real,Inv[0:400].real)
# xnew = np.linspace(0, 400, num=400)
plt.plot(time,Inv)
plt.legend(['data'],loc='best')
plt.show()








