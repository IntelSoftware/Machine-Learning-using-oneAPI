from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.decomposition import PCA
import numpy as np

############################# Import dpctl ######################
import dpctl
print(dpctl.__version__)
##################################################################


x = np.array([[1,1,1],[2,-1,3],[3,2,1]])

###################  Add code to get_devices, get_devices, select_gpu_device  ########
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
    
######################################################################################

            
            
############### Add code to convert x to dpctl.tensor x_device #########
x_device = dpctl.tensor.asarray(x, usm_type = 'device', device = "gpu")
######################################################################################


pca = PCA(2)  # 2 Principal components please

# replace x with x_device #######################################
est = pca.fit(x_device)
trans = pca.transform(x)  # replace trans with equivalent trans_x on device
trans_x = pca.transform(x_device)
################# Convert trans_x on GPU from dpctl.tensor to trans_host ########
trans_host =  dpctl.tensor.to_numpy(trans_x)
##################################################################

print('components_ ', pca.components_)
print('explained_variance_ ',pca.explained_variance_)

#### Choose the one that works, depending on your device!
# print('transformed x ',trans)
# print('transformed x ',trans_host)
print('PCA All Good\n')
