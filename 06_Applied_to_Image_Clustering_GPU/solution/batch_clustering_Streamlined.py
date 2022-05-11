# batch_clustering_Streamlined.py


#===============================================================================
# Copyright 2014-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

from sklearnex import patch_sklearn
patch_sklearn()

import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
from PIL.Image import Image as PilImage
#from skimage.color import rgb2hsv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import pandas as pd
import random
import operator
import seaborn as sns
import json
import dpctl

def ReshapeShortFat(original):
    """
    ReshapeShortFat(original)
    
    Reshapes the image numpy array from original shape 
    to the ShortFat single row shape and captures
    shape info in the output:
    channel_0, channel_1, channel_2
    
    functionally performs original.reshape(1, x*y*z) to create single row vector
    
    inputs:
        original - input the original image array
    
    return:
        ShortFatArray
        channel_0 - dimensions of the first channel - possibly the Red channel if using RGB
        channel_1 - dimensions of the first channel - possibly the Green channel if using RGB
        channel_2 - dimensions of the first channel - possibly the Blue channel if using RGB
    """       
    # preserve shape of incoming image
    channel_0, channel_1, channel_2 = original.shape
    #print('a shape ', a.shape)

    # convert to short, fat array
    ShortFatArray = original.reshape(1, channel_0*channel_1*channel_2)

    #print('a1 shape ', a1.shape)
    return ShortFatArray.squeeze(), channel_0, channel_1, channel_2

def Read_Transform_Images(resultsDict, 
                         imagesFilenameList = [], 
                         FSWRITE = False, path = '../03_Applied_to_Image_Clustering_CPU/data/'): 
    print('Running Read_Transform_Images on CPU: ')
    imageToClusterPath = path
    if len(imagesFilenameList) == 0:
        imagesFilenameList = [f for f in 
            sorted(glob.glob(imageToClusterPath 
            + '*.jpg'))]

    list_np_Images = []
    list_PIL_Images = []
    for im in imagesFilenameList:
        img =  Image.open(im)
        list_PIL_Images.append(img)
        a = np.asarray(img,dtype=np.float32)/255
        with np.errstate(divide='ignore', invalid='ignore'):
            #a1, x, y, z = ReshapeShortFat(rgb2hsv(a))
            a1, x, y, z = ReshapeShortFat(a)
        a2 = a1  # no image whitening for each file
        list_np_Images.append(a2)
    NP_images = np.array(list_np_Images)
    NP_images_STD = StandardScaler(with_std=True).fit_transform(NP_images)
    resultsDict['imagesFilenameList'] = imagesFilenameList
    resultsDict['list_PIL_Images'] = list_PIL_Images
    resultsDict['NP_images_STD'] = NP_images_STD
    if FSWRITE == True:
        write_results(resultsDict)
    return resultsDict

def displayImageGrid(img_arr, imageGrid=(4,5)):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np
    import random

    fig = plt.figure(figsize=(20,20))
    grid = ImageGrid(fig, 111, 
                     nrows_ncols=imageGrid,  # creates 2x2 grid of axes
                     #axes_pad=0.1,  # pad between axes
                     )

    img_arr = np.array(img_arr)
    for ax, im in zip(grid, img_arr):
         ax.imshow(im)
    plt.show()   
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def write_results_json(resultsDict):
    if 'list_PIL_Images' in resultsDict.keys():
        del resultsDict['list_PIL_Images']
    if 'NP_images_STD' in resultsDict.keys():
        del resultsDict['NP_images_STD']
    with open("results/resultsDict.json", "w") as outfile:
        json.dump(resultsDict, outfile, cls=NumpyEncoder)

def read_results_json():
    with open('results/resultsDict.json') as json_file:
        resultsDict = json.load(json_file)
    return resultsDict

def main():
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

    resultsDict = {}
    resultsDict = Read_Transform_Images(resultsDict)
    knee = 6   # number of clusters for KMeans
    EPS = 230  # EPS for dbscan
    n_components = 3  # nuber of components for PCA
    n_samples = 2     # number samples DBSCAN
    NP_images_STD = resultsDict['NP_images_STD'] # images as numpy array
    # Compute Follows Data: Determine if GPU is available
    if gpu_available:
        print('Running ComputePCA on GPU: ')   
        pca = PCA(n_components=n_components) # same as CPU
        
        ############### Difference bewteen CPU & GPU code ###################
        # convert data to dpctl.tensor prior to sklearnex call
        NP_images_STD_device = dpctl.tensor.from_numpy(NP_images_STD,  \
            usm_type = 'device', queue=dpctl.SyclQueue(device))
        #####################################################################    
        
        PCA_fit_transform = pca.fit_transform(NP_images_STD_device) # Use data/device on GPU
        k_means = KMeans(n_clusters = knee, init='random')
        db = DBSCAN(eps=EPS, min_samples = n_samples).fit(PCA_fit_transform) #use GPU data
        km = k_means.fit(PCA_fit_transform) #use GPU data
        # print types: any result you intend to pass around and use later
        # ensure it is converted to numpy prior to using it
        print("GPU: type(db.labels_)", type(db.labels_))
        print("GPU: type(km.labels_)", type(km.labels_))
        print("GPU: type(PCA_fit_transform)", type(PCA_fit_transform))
        
        ################  Cast returned dpctl.tensor._usmarray.usm_ndarray to numpy ###################
        print("GPU: type(dpctl.tensor.to_numpy(PCA_fit_transform))", type(dpctl.tensor.to_numpy(PCA_fit_transform)))
        ######################################################################################
        resultsDict['PCA_fit_transform'] = dpctl.tensor.to_numpy(PCA_fit_transform)
    else:
        print('Running ComputePCA on local CPU: ')
        device_context = 'cpu'
        pca = PCA(n_components=n_components)
        PCA_fit_transform = pca.fit_transform(NP_images_STD) 
        k_means = KMeans(n_clusters = knee, init='random')
        db = DBSCAN(eps=EPS, min_samples = n_samples).fit(PCA_fit_transform)
        km = k_means.fit(PCA_fit_transform) 
        resultsDict['PCA_fit_transform'] = PCA_fit_transform

    #resultsDict['device_context'] = device
    resultsDict['imageClusters_db'] = len(np.unique(db.labels_))
    resultsDict['counts_db'], resultsDict['bins_db']  = np.histogram(db.labels_, bins = EPS)
    resultsDict['counts'], resultsDict['bins'] =np.histogram(k_means.labels_, bins = knee)    
    resultsDict['imageClusters'] = len(np.unique(km.labels_))
    resultsDict['km_labels'] = km.labels_
    resultsDict['db_labels'] = db.labels_
    
    write_results_json(resultsDict)
    
    print("Kmeans bins   ", resultsDict['bins'])
    print("Kmeans counts ", resultsDict['counts'])
    
    print("All good inside main\n")
    
    return resultsDict

if __name__ == "__main__":
    resultsDict = main()
    #print("km_list: ", resultsDict['km_list'][0:2])
    print('All looks good!\nRun 03_Plot_GPU_Results.ipynb to graph the results!')

# Notices & Disclaimers 

# Intel technologies may require enabled hardware, software or service activation.
# No product or component can be absolutely secure.

# Your costs and results may vary.

# © Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. 
# *Other names and brands may be claimed as the property of others.
