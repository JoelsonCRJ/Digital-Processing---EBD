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

        global M,L
        
        for i in range(0,len(labels)):
                v=np.array([])
                intervalo_inicial = intervalos[i][0]
                intervalo_final = intervalos[i][1]
                for j in range(0,11):
                        v=np.append(v,musc[j][intervalo_inicial:intervalo_final])
                        M=np.concatenate((M,v),axis=0)        
        
sessions = load_dict('/home/joelson/Documents/Digital-Processing-EBD-UFES-2019.1/02-EMG/data.pkl')
#plot_session(sessions['1'][0].T[0:11],sessions['1'][1],sessions['1'][2],M,L)

v=np.array(sessions['1'][0].T[0:11][0:800])

M=np.array([])
L=np.array([])

descript_vectors(sessions['1'][0].T[0:11],sessions['1'][1],sessions['1'][2])
print(M.shape)
# v.flatten()
# print(v)
# a=(sessions['1'][0:11][0:800])



