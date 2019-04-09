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

def interp_peaks(peaks_index_array,array,window_size): 
    for i in range (0,len(peaks_index_array)):
        #para picos com enderecos menores que a janela utilizar uma janela maior a direita 
        #com polinomio interpolador
        if (peaks_index_array[i]<window_size): 
            x = np.arange(0,(peaks_index_array[i]+6*window_size))
            y=array[(peaks_index_array[i]+6*window_size):(peaks_index_array[i]+(2*6*window_size)+1)]
            f=interp1d(x,y,kind='cubic')
            array[0:peaks_index_array[i]+6*window_size]=f(x)
        
        elif (array[peaks_index_array[i]].real>15):
            x = np.arange((peaks_index_array[i]-60),(peaks_index_array[i]+60))
            y=array[(peaks_index_array[i]-(60+120)):(peaks_index_array[i]-60)]
            f=interp1d(x,y,kind='cubic')
            array[(peaks_index_array[i]-60):(peaks_index_array[i]+60)] = f(x)
        else:
            x = np.arange((peaks_index_array[i]-180),(peaks_index_array[i]+180))
            y=array[(peaks_index_array[i]-(180+360)):(peaks_index_array[i]-180)]
            f=interp1d(x,y,kind='cubic')
            array[(peaks_index_array[i]-180):(peaks_index_array[i]+180)] = f(x)
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
#codigo para plotar fft
# P2=abs(Y/N)
# a=int ((N/2)+1)
# P1=P2[2:a]
# P1[2:N-1] = 2*P1[2:N-1]
# f = (Fs/N)*np.arange(0,(N/2),1)
# plt.plot(f[0:len(f)-1],P1)-1
# plt.show()

Inv = scipy.ifft(Y)
ind = detect_peaks(Inv,mph=0, mpd=(750),show=True)
Inv=interp_peaks(ind,Inv,50)
print(ind)


plt.figure(1)

plt.subplot(211)
plt.plot(time,data,'r')
plt.legend(['EMG +ECG'],loc='best')
plt.ylim(440, 560)
plt.xlim(0,60000)

plt.subplot(212)
plt.plot(time,Inv)
plt.legend(['EMG'],loc='best')
plt.ylim(-60, 60)
plt.xlim(0,60000)
plt.show()








