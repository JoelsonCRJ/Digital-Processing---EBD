from sklearn import decomposition
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import scipy as sp
from scipy import signal
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
from six.moves import cPickle as pickle
from sklearn.decomposition import PCA

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def plot_session(musculos, intervalos, labels):

    colors=['#29ffa2', '#2969ff', '#ff9029', '#ff2929', '#fbff29', '#000000']
    fig, axes = plt.subplots(len(musculos), 1)
    for j in range(len(musculos)):
        
        axes[j].plot([x for x in range(musculos[j].size)], musculos[j], 'b-', linewidth=0.8)
        for i in range(len(intervalos)):
             axes[j].axvline(intervalos[i][0], color='#000000',alpha=0.5, linewidth=1.5)
             axes[j].text(intervalos[i][0], 0, str(int(labels[i])), fontsize = 8)
             axes[j].axvline(intervalos[i][1]-5, color='#000000',alpha=0.5, linewidth=1.5)
    plt.show()


def descript_vectors(musc, intervalos,labels):
        M=np.array([])
        L=np.array([])
        global V1_training,V2_training,V3_training,V4_training,V5_training,V6_training
        for i in range(0,len(labels)):
                v=np.array([])
                intervalo_inicial = intervalos[i][0]
                intervalo_final = intervalos[i][1]
                for j in range(0,11):
                        v=np.append(v,musc[j][intervalo_inicial:intervalo_final])
                if(labels[i]==1):
                        V1_training=np.concatenate((V1_training,v),axis=0)
                elif(labels[i]==2):
                        V2_training=np.concatenate((V2_training,v),axis=0)
                elif(labels[i]==3):
                        V3_training=np.concatenate((V3_training,v),axis=0)
                elif(labels[i]==4):
                        V4_training=np.concatenate((V4_training,v),axis=0)
                elif(labels[i]==5):
                        V5_training=np.concatenate((V5_training,v),axis=0)
                else:
                        V6_training=np.concatenate((V6_training,v),axis=0)
                # M=np.concatenate((M,v),axis=0)   
                # L=np.append(L,labels[i])
                #print((M.size)/800)
        # M=M.reshape(44,8800)
        # return M,L
        


sessions = load_dict('/home/familia/Documents/Digital-Processing-EBD-UFES-2019.1/02-EMG/data.pkl')
#plot_session(sessions['1'][0].T[0:11],sessions['1'][1],sessions['1'][2])


V1_training=np.array([])
V2_training=np.array([])
V3_training=np.array([])
V4_training=np.array([])
V5_training=np.array([])
V6_training=np.array([])



#treinamento
for i in range(1,10):
        descript_vectors(sessions['{}'.format(i)][0].T[0:11],sessions['1'][1],sessions['1'][2])

description_vector_size=8800
V1_training=V1_training.reshape(int((V1_training.size)/description_vector_size),description_vector_size)
V2_training=V2_training.reshape(int((V2_training.size)/description_vector_size),description_vector_size)
V3_training=V3_training.reshape(int((V3_training.size)/description_vector_size),description_vector_size)
V4_training=V4_training.reshape(int((V4_training.size)/description_vector_size),description_vector_size)
V5_training=V5_training.reshape(int((V5_training.size)/description_vector_size),description_vector_size)
V6_training=V6_training.reshape(int((V6_training.size)/description_vector_size),description_vector_size)


#aplicando PCA
print(V4_training.shape)





        

# v.flatten()
# print(v)
# a=(sessions['1'][0:11][0:800])



