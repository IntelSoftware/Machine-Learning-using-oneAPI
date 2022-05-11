
# this demonstrates a sequence of calls, fit followed by predict
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np
import dpctl

def compute_fit_then_predict_gpu(gpu_device):
    
    import os
    
    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn import datasets
    from sklearn.cluster import KMeans
    print("\nComputing using compute_fit_then_predict_gpu\n")
    nCentroids = 3

    iris = datasets.load_iris()
    columns = iris.feature_names
    data_host = iris.data

    # move data to dpctl tensor on gpu
    data_device = dpctl.tensor.from_numpy(data_host, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))  
    kmeans = KMeans(nCentroids, init='random', random_state=0)
    
    # **************************************************************
    # this demonstrates a sequence of calls, fit followed by predict    
    kmeans_model = kmeans.fit(data_device) 
    
    #*** Persisting a model for later use ***
    from joblib import dump    
    dump(kmeans_model, 'lab/kmeans_model.joblib') #dump model
    dump(data_host, 'data/data_host.joblib') # dump X values if desired
    #*****************************************
    # OPTIONAL, 
    #   in real applications, training is done at a different time, place and inference machine
    #   so a mechanism to save and retreive a model is requred
    #   OPTIONAL, store the model to disk: loading pickle files form disk is a security issue
    #   consult documentation from scikit-learn to learn more options    
    
    #*** Reloading a Persisted model ***
    from joblib import load
    kmeans_recovered_model = load( 'lab/kmeans_model.joblib')
    data_host = load( 'data/data_host.joblib')
    result_device = kmeans_recovered_model.predict(data_device)
    print('type(result_device)', type(result_device)) # 
    #*******************************************************
    # move data from device dpctl tesnor to host    
    result_npy = dpctl.tensor.to_numpy(result_device)
    #*******************************************************
    print("kmeans.get_params: ",kmeans.get_params)
    labels = pd.DataFrame( result_npy, columns = ['labels'] )

    labels.to_csv('kmeans_Labels.csv', index=False)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_)
    cluster_centers.to_csv('kmeans_ClusterCenters.csv', index=False )

gpu_available = False
for d in dpctl.get_devices():
    if d.is_gpu:
        #print("GPU Found")
        gpu_device = dpctl.select_gpu_device()
        gpu_available = True
    else:
        #print("Default to CPU")
        cpu_device = dpctl.select_cpu_device() 
if gpu_available:
    print("GPU available: ", gpu_device.name)
    device = gpu_device
else:
    print("CPU available: ", cpu_device.name)
    device = cpu_device

if gpu_available:
    import time
    print("gpu_available ", gpu_available)
    print("Device Targeted: ", device.name)
    start = time.time()
    compute_fit_then_predict_gpu(device)
    print('Execution time: ', time.time() - start)
    cluster_centers = pd.read_csv('kmeans_ClusterCenters.csv')
    labels = pd.read_csv('kmeans_Labels.csv')
    print(labels)
else:
    print("\n****\ntry again - no GPU available\n*****")
