from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.decomposition import PCA
import numpy as np

############################# Import dpctl ######################

##################################################################


x = np.array([[1,1,1],[2,-1,3],[3,2,1]])

            ###################  Add code to get_devices, get_devices, select_gpu_device  ########

            ######################################################################################

            
            
            ############### Add code to convert x to dpctl.tensor x_device #########

            
            ######################################################################################


pca = PCA(2)  # 2 Principal components please

################# Convert x to dpctl.tensor accessing gpu ########
#  x_device = __________
##################################################################

# repalce x with x_device #######################################
est = pca.fit(x)
trans = pca.transform(x)  # replace trans with equivalent trans_x on device
# trans_x = pca.transform(x)
################# Convert trans_x on GPU from dpctl.tensor to trans_host ########
# trans_host =  ______________
##################################################################

print('components_ ', pca.components_)
print('explained_variance_ ',pca.explained_variance_)

#### Choose the one that works, depending on your device!
# print('transformed x ',trans)
# print('transformed x ',trans_host)
print('PCA All Good\n')
