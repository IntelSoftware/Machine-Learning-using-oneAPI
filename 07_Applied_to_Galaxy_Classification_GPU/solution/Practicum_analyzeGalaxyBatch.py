# Comment the writefile statement in order to run this cell immediately, #comment it out to write the file for targeted execution on another device
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()
import dpctl

import warnings
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected")


from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
from random import sample
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

XenoSupermanGalaxy_Arms = np.load('../Data/XenoSupermanGalaxy_Arms.npy')
XenoSupermanGalaxy_CenterGlob = np.load('../Data/XenoSupermanGalaxy_CenterGlob.npy')
XenoSupermanGalaxy_Stars = np.load('../Data/XenoSupermanGalaxy_Stars.npy')
XenoSupermanGalaxy_tooClose = np.load('../Data/XenoSupermanGalaxy_tooClose.npy')

XenoSupermanGalaxy = {}
XenoSupermanGalaxy['Arms'] = XenoSupermanGalaxy_Arms
XenoSupermanGalaxy['CenterGlob'] = XenoSupermanGalaxy_CenterGlob
XenoSupermanGalaxy['Stars'] = XenoSupermanGalaxy_Stars
XenoSupermanGalaxy['tooClose'] = XenoSupermanGalaxy_tooClose

GFFA_Arms = np.load('../Data/GFFA_Arms.npy')
GFFA_CenterGlob = np.load('../Data/GFFA_CenterGlob.npy')
GFFA_Stars = np.load('../Data/GFFA_Stars.npy')
GFFA_tooClose = np.load('../Data/GFFA_tooClose.npy')

GFFA = {}
GFFA['Arms'] = GFFA_Arms
GFFA['CenterGlob'] = GFFA_CenterGlob
GFFA['Stars'] = GFFA_Stars
GFFA['tooClose'] = GFFA_tooClose

plt.style.use('dark_background')
    
# dataset is subset of stars from each galaxy
TrainingSize = min(len(GFFA['Stars']), len(XenoSupermanGalaxy['Stars'] ) ) 

collision = dict()
collision['Arms'] = np.vstack((GFFA['Arms'].copy(), XenoSupermanGalaxy['Arms'].copy()))
collision['CenterGlob'] = np.vstack((GFFA['CenterGlob'].copy(), XenoSupermanGalaxy['CenterGlob'].copy()))
collision['Stars'] = np.vstack((GFFA['Stars'].copy(), XenoSupermanGalaxy['Stars'].copy()))
collision['Stars'].shape

# get the index of the stars to use from XenoSupermanGalaxy
XenoIndex = np.random.choice(len(XenoSupermanGalaxy['Stars']), TrainingSize, replace=False)
# get the index of the stars to use from GFFAIndex
GFFAIndex = np.random.choice(len(GFFA['Stars']), TrainingSize, replace=False)

# create a list with a labelforeahc item in the combined training set
# the first hald of the list indicates that class 0 will be for GFFA, 1 will be XenoSupermanGalaxy
y = [0]*TrainingSize + [1]*TrainingSize
# Stack the stars subset in same order as the labels, GFFA first, XenoSupermanGalaxy second
trainGalaxy = np.vstack((GFFA['Stars'][GFFAIndex], XenoSupermanGalaxy['Stars'][XenoIndex]))  

x_train, x_test, y_train, y_test = train_test_split(trainGalaxy, np.array(y), train_size=0.05)


K = 3
myModels = {'KNeighborsClassifier':KNeighborsClassifier(n_neighbors = K) , 
            'RandomForestClassifier': RandomForestClassifier(n_jobs=2, random_state=0), 
           }
#sweep through various training split percentage
TrainingSize = [.001, .01, .03, .05, .1, .2, .5, .8]
bestScore = {}
hi = 0
K = 3

# Compute Follows Data
# determine if GPU available:
for d in dpctl.get_devices(): #loop thru all available devices
    gpu_available = False
    for d in dpctl.get_devices():  # Note if GPU is found
        if d.is_gpu:
            gpu_device = dpctl.select_gpu_device()  # get device context
            gpu_available = True
        else:
            cpu_device = dpctl.select_cpu_device()  # get device context
if gpu_available:
    device = gpu_device
    print("GPU targeted: ", device)
else:
    device = cpu_device
    print("CPU targeted: ", device)
        
for tsz in TrainingSize:
    x_train, x_test, y_train, y_test = train_test_split( \
                trainGalaxy, np.array(y), train_size=tsz)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    x_train_device = dpctl.tensor.from_numpy(x_train,  \
            usm_type = 'device', queue=dpctl.SyclQueue(device))
    y_train_device = dpctl.tensor.from_numpy(y_train,  \
            usm_type = 'device', queue=dpctl.SyclQueue(device))  
    x_test_device = dpctl.tensor.from_numpy(x_test,  \
            usm_type = 'device', queue=dpctl.SyclQueue(device)) 
    y_test_device = dpctl.tensor.from_numpy(y_test,  \
            usm_type = 'device', queue=dpctl.SyclQueue(device))  
    
    for name, modelFunc in myModels.items():   
        print("Compute Device: ", device)
        start = time.time()
        model = modelFunc

        model.fit(x_train_device, y_train_device)
        y_pred_device = model.predict(x_test_device) 
        y_pred = dpctl.tensor.to_numpy(y_pred_device)
        
        print('Results of {} classification'.format(name))
        print('  K: ',K)
        print('  Training size: ', tsz)
        print('  y_train.shape: ',y_train.shape)
        roc = roc_auc_score(y_test, y_pred)
        print('  roc_auc_score: {:4.1f}'.format(100*roc))
        print('  Time: {:5.1f} sec\n'.format( time.time() - start))
        if roc > hi:
            hi = roc
            bestScore = {'name': name,
                    'roc':roc, 
                    'trainingSize':tsz, 
                    'confusionMatrix': confusion_matrix(y_test, y_pred), 
                    'precision': 100*precision_score(y_test, y_pred, average='binary'),
                    'recall': recall_score(y_test, y_pred, average='binary') }
print('bestScore: name', bestScore['name'])
print('bestScore: confusion Matrix', bestScore['confusionMatrix'])
print('bestScore: precision', bestScore['precision'])
print('bestScore: recall', bestScore['recall'])
print('bestScore: roc', bestScore['roc'])

# Notices & Disclaimers

# Intel technologies may require enabled hardware, software or service activation.

# No product or component can be absolutely secure.
# Your costs and results may vary.

# © Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. 
# *Other names and brands may be claimed as the property of others.
