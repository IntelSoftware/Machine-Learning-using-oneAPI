from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.decomposition import PCA
import numpy as np
import dpctl

x = np.array([[1,1,1],[2,-1,3],[3,2,1]])

for d in dpctl.get_devices():
    gpu_available = False
    for d in dpctl.get_devices():
        if d.is_gpu:
            gpu_device = dpctl.select_gpu_device()
            gpu_available = True
        else:
            cpu_device = dpctl.select_cpu_device() 
if gpu_available:
    print("GPU targeted: ", gpu_device)
else:
    print("CPU targeted: ", cpu_device)

pca = PCA(2)

x_device = dpctl.tensor.from_numpy(x, usm_type = 'device', device = dpctl.SyclDevice("gpu"))
est = pca.fit(x_device)
trans_x = pca.transform(x_device)
trans_host =  dpctl.tensor.to_numpy(trans_x)
print('components_ ', pca.components_)
print('explained_variance_ ',pca.explained_variance_)
print('transformed x ',trans_host)
print('PCA All Good\n')
