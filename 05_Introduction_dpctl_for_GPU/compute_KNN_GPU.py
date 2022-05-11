# Copyright 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import pandas as pd
import dpctl
from sklearnex import patch_sklearn
patch_sklearn()

from  sklearn.datasets import fetch_covtype
x, y = fetch_covtype(return_X_y=True)
# Data Set Information:
# Predicting forest cover type from cartographic variables only (no remotely sensed data). The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables (wilderness areas and soil types).
# This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

# for sake of time is 1/4th of the data
subset = x.shape[0]//4
x = x[:subset,:]
y = y[:subset]

# Is this computed on GPU or on Host? Remember compute follows data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=72)

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

if gpu_available:
    # target a remote hosy CPU when submitted via q.sh or qsub -I
    x_train_device = dpctl.tensor.from_numpy(x_train, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))
    y_train_device = dpctl.tensor.from_numpy(y_train, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))    
    x_test_device = dpctl.tensor.from_numpy(x_test, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))
    y_test_device = dpctl.tensor.from_numpy(y_test, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))
else:
    # target a remote hosy CPU when submitted via q.sh or qsub -I
    x_train_device = dpctl.tensor.from_numpy(x_train, usm_type = 'device', queue=dpctl.SyclQueue(cpu_device))
    y_train_device = dpctl.tensor.from_numpy(y_train, usm_type = 'device', queue=dpctl.SyclQueue(cpu_device))    
    x_test_device = dpctl.tensor.from_numpy(x_test, usm_type = 'device', queue=dpctl.SyclQueue(cpu_device))
    y_test_device = dpctl.tensor.from_numpy(y_test, usm_type = 'device', queue=dpctl.SyclQueue(cpu_device))    

# set up KNN algorithm parameters
# 'n_neighbors': 40,  
#     regulates how many neighbors should be checked when an item is being classified
# 'weights': 'distance',
#     signifies how weight should be distributed between neighbor values.
#     This value will cause weights to be distributed based on their distance (inversely correlated). Closer neighbors will have a higher weight in the algorithm.
# 'n_jobs': -1
#     Signifies the parallel jobs to be allowed at the same time for neighbor algorithm
params = {
    'n_neighbors': 40,  
    'weights': 'distance'
}
print('dataset shape: ', x_train_device.shape)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(**params).fit(x_train_device, y_train_device)
predictedGPU = knn.predict(x_test_device) #Predict on GPU
predictedCPU = knn.predict(x_test) #Predict on CPU

report = metrics.classification_report(y_test, predictedCPU)
print(f"Classification report for kNN:\n{report}\n")
