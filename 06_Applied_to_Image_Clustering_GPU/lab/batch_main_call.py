

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
import os

def main():
    # determine if GPU available:
    gpu_available = False
    cpu_available = False

    # determine which device control method is appropriate for my server and software stack, and determine if cpu or gpu is available
    from daal4py.oneapi import sycl_context
    resultsDict = {}
    try:
        with sycl_context('gpu'):
            gpu_available = True
            device_context = 'gpu'
            resultsDict['device_context'] = device_context
            np.save('results/device_context', device_context)
            def gpu_context():
                return sycl_context('gpu')
    except:
         with sycl_context('cpu'):
            cpu_available = True
            device_context = 'cpu'
            resultsDict['device_context'] = device_context
            np.save('results/device_context', device_context)
            def cpu_context():
                return sycl_context('cpu')           

    print('gpu_available: ', gpu_available)
    print('cpu_available: ', cpu_available)
    
    imagesFilenameList = np.load('results/resultsDict_imagesFilenameList.npy')
    
    resultsDict = Read_Tansform_Images(resultsDict,imagesFilenameList = imagesFilenameList)
    
    resultsDict = ComputePCA(resultsDict, n_components = 6, gpu_available = gpu_available)
    del resultsDict['NP_images_STD']
    
    resultsDict = Compute_kmeans_inertia(resultsDict, gpu_available = gpu_available)
    knee = 6
    EPS = 350
    resultsDict = Compute_kmeans_db_histogram_labels(resultsDict,  knee = knee, EPS = EPS, gpu_available = gpu_available) 

    write_results(resultsDict)
    
    print(resultsDict['bins'])
    print(resultsDict['counts'])
    
    print("All good inside main\n")
    
    return resultsDict

if __name__ == "__main__":
    resultsDict = main()
    #print("km_list: ", resultsDict['km_list'][0:2])
    print('All looks good!\nRun 03_Plot_GPU_Results.ipynb to graph the results!')
