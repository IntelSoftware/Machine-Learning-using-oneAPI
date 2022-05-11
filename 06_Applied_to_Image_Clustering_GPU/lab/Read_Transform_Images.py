
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

# define contexts under which to run
# try:
#     from dpctx import device_context, device_type

#     def gpu_context():
#         return device_context(device_type.gpu, 0)

#     def cpu_context():
#         return device_context(device_type.cpu, 0)
# except:
#     from daal4py.oneapi import sycl_context

#     def gpu_context():
#         return sycl_context('gpu')

#     def cpu_context():
#         return sycl_context('cpu')
        
def ReshapeShortFat(original):
    """
    ReshapeShortFat(original)
    
    Reshapes the image numpy array from original shape 
    to the ShortFat single row shape and captures
    shape info in the output:
    channel_0, channel_1, channel_2
    
    functionally performs original.reshape(1, x*y*z) to create single row vector
    
    inputs:
        original - input the oringal image array
    
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
                         FSWRITE = False, path = 'data/'): 
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
        # standardize - mean =o, std = 1
        #a2 = StandardScaler(with_std=True).fit_transform(a1.reshape(-1, 1)).flatten() #whiten each image
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
    
def displayImageGrid2(img_arr, ncols=8):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from PIL import Image

    def img_reshape(img):
        img = Image.open('./data/'+img).convert('RGB')
        img = img.resize((300,300))
        img = np.asarray(img)
        return img

    def image_grid(array, ncols=ncols):
        print(array.shape)
        index, height, width, channels = array.shape
        nrows = index//ncols

        img_grid = (array.reshape(nrows, ncols, height, width, channels)
                  .swapaxes(1,2)
                  .reshape(height*nrows, width*ncols, 3))
        return img_grid
    
    result = image_grid(np.array(img_arr))
    fig = plt.figure(figsize=(20., 20.))
    plt.imshow(result)


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
        
# def write_results(resultsDict):
#     print("write_results...")
#     np.save('results/device_context', resultsDict['device_context'])
#     np.save('results/resultsDict_imagesFilenameList', resultsDict['imagesFilenameList'])
#     np.save('results/resultsDict_counts', resultsDict['counts'])
#     np.save('results/resultsDict_bins', resultsDict['bins'])
#     np.save('results/resultsDict_imageClusters', resultsDict['imageClusters'])
#     np.save('results/resultsDict_model_labels', resultsDict['model'].labels_)
#     np.save('results/resultsDict_counts_db', resultsDict['counts_db'])
#     np.save('results/resultsDict_bins_db', resultsDict['bins_db'])
#     np.save('results/resultsDict_imageClusters_db', resultsDict['imageClusters_db'])
#     np.save('results/resultsDict_model_labels_db', resultsDict['model_db'].labels_)
#     #np.save('results/resultsDict_list_PIL_Images', resultsDict['list_PIL_Images'])
#     pd.DataFrame.from_dict(resultsDict['km_list']).to_csv('results/km_list.csv')
#     np.save('results/resultsDict_PCA_fit_transform', resultsDict['PCA_fit_transform'])
#     print("write complete...")
    

# def read_results():
#     resultsDict = {}
#     resultsDict['device_context'] = np.load('results/device_context.npy', allow_pickle = True)
#     resultsDict['imagesFilenameList'] = np.load('results/resultsDict_imagesFilenameList.npy', allow_pickle = True)
#     resultsDict['counts'] = np.load('results/resultsDict_counts.npy', allow_pickle = True)
#     resultsDict['bins'] = np.load('results/resultsDict_bins.npy', allow_pickle = True)
#     resultsDict['imageClusters'] = np.load('results/resultsDict_imageClusters.npy', allow_pickle = True)
#     resultsDict['model'] = {}
#     resultsDict['model']['labels_'] =  np.load('results/resultsDict_model_labels.npy', allow_pickle = True)
    
#     resultsDict['counts_db'] = np.load('results/resultsDict_counts_db.npy', allow_pickle = True)
#     resultsDict['bins_db'] = np.load('results/resultsDict_bins_db.npy', allow_pickle = True)
#     resultsDict['imageClusters_db'] = np.load('results/resultsDict_imageClusters_db.npy', allow_pickle = True)
#     resultsDict['model_db'] = {}
#     resultsDict['model_db']['labels_'] =  np.load('results/resultsDict_model_labels_db.npy', allow_pickle = True)
    
    #resultsDict['km_list'] = pd.read_csv('results/km_list.csv')
    
    resultsDict['PCA_fit_transform'] = np.load('results/resultsDict_PCA_fit_transform.npy', allow_pickle = True)

    print("read complete...")
    return resultsDict

