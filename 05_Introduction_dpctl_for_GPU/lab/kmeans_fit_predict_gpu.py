
# this demonstrates using fit_predict on GPU or CPU
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np
import dpctl
import time

def compute_fit_predict_gpu(gpu_device):
    
    import os
    
    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn import datasets
    from sklearn.cluster import KMeans
    print("\n Computing using compute_fit_predict_gpu")

    nCentroids = 3

    iris = datasets.load_iris()
    columns = iris.feature_names
    data = iris.data

    # move data to dpctl tensor on gpu
    data = dpctl.tensor.from_numpy(data, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))
    print('type(data)', type(data))
    kmeans = KMeans(nCentroids, init='random', random_state=0)
    
    # **************************************************************    
    result = kmeans.fit_predict(data)
    print('type(result)', type(result))
    # **************************************************************  
    
    print('type(kmeans.cluster_centers_)', type(kmeans.cluster_centers_))
    # move data from device dpctl tesnor to host
    labels = dpctl.tensor.to_numpy(result)
    labels = pd.DataFrame( labels, columns = ['labels'] )

    labels.to_csv('kmeans_Labels.csv', index=False)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_)
    cluster_centers.to_csv('kmeans_ClusterCenters.csv', index=False )

gpu_available = False
for d in dpctl.get_devices():
    if d.is_gpu:
        gpu_device = dpctl.select_gpu_device()
        gpu_available = True
    else:
        cpu_device = dpctl.select_cpu_device() 
if gpu_available:
    print("GPU available: ", gpu_device.name)
else:
    print("CPU available: ", cpu_device.name)

if gpu_available:
    import time
    device = gpu_device
    print("Device Targeted: ", device.name)
    start = time.time()
    compute_fit_predict_gpu(device)
    print('Execution time: ', time.time() - start)
    cluster_centers = pd.read_csv('kmeans_ClusterCenters.csv')
    labels = pd.read_csv('kmeans_Labels.csv')
    print(labels)
else:
    print("\n****\ntry again - no GPU available\n*****")
