from scipy import io
import matplotlib.pyplot as plt
import numpy as np

inMATFile = 'S02.mat'
data = io.loadmat(inMATFile)

print(data.keys())

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
# emg_signal_6 = np.array((data_EMG_signals[:,5]))
# emg_signal_7 = np.array((data_EMG_signals[:,6]))
# emg_signal_8 = np.array((data_EMG_signals[:,7]))
# emg_signal_9 = np.array((data_EMG_signals[:,8]))
# emg_signal_10 = np.array((data_EMG_signals[:,9]))


#ang=((3/(pow(2,16)-1))*(angulos-(3/2)))/((3/2)*606*(pow(10,-5)))
#print(len(data_knee_angle))
#print(ang)



# for i in range(0,len(classes)):
#     if(classes[i]==1):
#         tempo_inicial[i]
#         tempo_final[i]
#plot canais
# print(len(emg_signal_1
# ))
plt.subplot(10,1,1)
plt.plot(tempo,emg_signal_1)

plt.subplot(10,1,2)
plt.plot(tempo,emg_signal_2)

plt.subplot(10,1,3)
plt.plot(tempo,emg_signal_3)

plt.subplot(10,1,4)
plt.plot(tempo,emg_signal_4)

plt.subplot(10,1,5)
plt.plot(tempo,emg_signal_5)

plt.subplot(10,1,6)
plt.plot(tempo,data_knee_angle)

# plt.subplot(10,1,6)
# plt.plot(tempo,emg_signal_6)

# plt.subplot(10,1,7)
# plt.plot(tempo,emg_signal_7)

# plt.subplot(10,1,8)
# plt.plot(tempo,emg_signal_8)

# plt.subplot(10,1,9)
# plt.plot(tempo,emg_signal_9)

# plt.subplot(10,1,10)
# plt.plot(tempo,emg_signal_10)

# #plt.suptitle(str(classes))

plt.show()


