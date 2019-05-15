from scipy import io
import matplotlib.pyplot as plt
import numpy as np

inMATFile = 'S02.mat'
data = io.loadmat(inMATFile)

#print(data.keys())

data_Fs = data['Fs']
data_class_mapf=data['class_mapf']
data_EMG_signals= data['EMG_signals']
data_knee_angle=data['knee_angle']
data_sEMG_labels=data['sEMG_labels']

#print(data_EMG_signals)

secoes = np.array(data_class_mapf[:,0])
classes=np.array(data_class_mapf[:,1])
tempo_inicial=np.array(data_class_mapf[:,2])
tempo_final=np.array(data_class_mapf[:,3])
# print(secoes[21])
# print(tempo_final[21])
tempo=np.arange(0,1260441,1)
emg_signal_1 = np.array((data_EMG_signals[:,0]))
emg_signal_2 = np.array((data_EMG_signals[:,1]))
emg_signal_3 = np.array((data_EMG_signals[:,2]))
emg_signal_4 = np.array((data_EMG_signals[:,3]))
emg_signal_5 = np.array((data_EMG_signals[:,4]))
emg_signal_6 = np.array((data_EMG_signals[:,5]))
emg_signal_7 = np.array((data_EMG_signals[:,6]))
emg_signal_8 = np.array((data_EMG_signals[:,7]))
emg_signal_9 = np.array((data_EMG_signals[:,8]))
emg_signal_10 = np.array((data_EMG_signals[:,9]))


print(emg_signal_10)
secoes_vector = np.zeros(len(tempo))
secoes_classe = np.zeros(len(tempo))
print(len(classes))
for i in range(0,20):
    if(i>=0 and i<=9):
        secoes_vector[tempo_inicial[i*22]:tempo_final[(i*22)+21]]= i+1
    elif(i>=10 and i<=20):
        secoes_vector[tempo_inicial[i*22]:tempo_final[(i*22)+21]]= ((i+1)-10)

for j in range(0,440):
    secoes_classe[tempo_inicial[j]:tempo_final[j]] = classes[j]



plt.figure(1)
#plot canais

plt.subplot(13,1,1)
plt.plot(tempo,secoes_vector)
plt.xlim(0,1260450)

plt.subplot(13,1,2)
plt.plot(tempo,secoes_classe,'r')
plt.xlim(0,1260450)

plt.subplot(13,1,3)
plt.plot(tempo,data_knee_angle)
plt.xlim(0,1260450)

plt.subplot(13,1,4)
plt.plot(tempo,emg_signal_1)
plt.xlim(0,1260450)

plt.subplot(13,1,5)
plt.plot(tempo,emg_signal_2)
plt.xlim(0,1260450)

plt.subplot(13,1,6)
plt.plot(tempo,emg_signal_3)
plt.xlim(0,1260450)

plt.subplot(13,1,7)
plt.plot(tempo,emg_signal_4)
plt.xlim(0,1260450)

plt.subplot(13,1,8)
plt.plot(tempo,emg_signal_5)
plt.xlim(0,1260450)

plt.subplot(13,1,9)
plt.plot(tempo,emg_signal_6)
plt.xlim(0,1260450)

plt.subplot(13,1,10)
plt.plot(tempo,emg_signal_7)
plt.xlim(0,1260450)

plt.subplot(13,1,11)
plt.plot(tempo,emg_signal_8)
plt.xlim(0,1260450)

plt.subplot(13,1,12)
plt.plot(tempo,emg_signal_9)
plt.xlim(0,1260450)

plt.subplot(13,1,13)
plt.plot(tempo,emg_signal_10)
plt.xlim(0,1260450)
# #plt.suptitle(str(classes))


plt.show()



