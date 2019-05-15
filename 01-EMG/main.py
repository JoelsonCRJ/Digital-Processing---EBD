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
YN=scipy.fftpack.fft(data)
#codigo para plotar fft data
P2N=abs(YN/N)
an=int ((N/2)+1)
P1N=P2N[2:an]
P1N[2:N-1]=2*P1N[2:N-1]
fN = (Fs/N)*np.arange(0,(N/2),1)
plt.figure(1)

plt.xlabel('Frequência (Hz)')
plt.ylabel('|EMG(t)|')
plt.plot(fN[0:len(fN)-1],P1N)
plt.legend(['FFT do sinal EMG antes da filtragem'],loc='best')
###################################
Y=scipy.fftpack.fft(Y_butter)
#codigo para plotar fft butter
P2=abs(Y/N)
a=int ((N/2)+1)
P1=P2[2:a]
P1[2:N-1] = 2*P1[2:N-1]
f = (Fs/N)*np.arange(0,(N/2),1)
plt.figure(2)

plt.plot(f[0:len(f)-1],P1,'r')
plt.legend(['FFT do sinal EMG após filtragem'],loc='best')
plt.xlabel('Frequência (Hz)')
plt.ylabel('|EMG*(t)|')
plt.show()



a=scipy.ifft(Y)
Inv = scipy.ifft(Y)
ind = detect_peaks(Inv,mph=0, mpd=(750),show=None)
Inv=interp_peaks(ind,Inv,50)
#print(ind)


# plt.figure(1)
# plt.plot(time,data,'b')
# plt.legend(['Sinal original'],loc='best')
# plt.ylim(440, 560)
# plt.xlim(0,60000)
# plt.xlabel('tempo (ms)')
# plt.ylabel('Valor de sinal quantizado - 12bits')

# plt.figure(2)
# plt.plot(time,a,'r')
# plt.legend(['Sinal filtrado - Butter 5ª ordem'],loc='best')
# plt.ylim(-60, 60)
# plt.xlim(0,60000)

# plt.xlabel('tempo (ms)')
# plt.ylabel('Valor de sinal quantizado - 12bits')


plt.figure(3)
plt.plot(time,a,'b',label='Sinal filtrado com ECG')
plt.plot(time,Inv,'r',label='Sinal filtrado sem ECG')
plt.legend(['Sinal filtrado com ECG','Sinal filtrado sem ECG'],loc='best')
plt.xlabel('tempo (ms)')
plt.ylabel('Valor de sinal quantizado - 12bits')
plt.ylim(-60, 60)
plt.xlim(0,60000)

plt.show()








