#!/usr/bin/env python
# coding: utf-8

# # Module 01_01: Classifier: targeting CPU and Patching 
# 
# ![Assets/KNNacceleration.jpg](Assets/KNNacceleration.jpg)
# ### Use nbconvert  patch_sklearn from command line

# # Learning Objectives:
# 
# 1) Describe how to surgically unpatch specific optimized functions if needed
# 1) Apply patching to KNN algorithm
# 2) Describe acceleration for the covtype dataset with KNN classification
# 
# 

# # *Real World* example Classifier on CovType Dataset
# 
# ### Compare timings of stock kmeans versus Intel Extension for Scikit-learn Classifier using patch_sklean()
# 
# Below we will apply Intel Extension for Scikit learn to a use case on a CPU
# 
# Intel® Extension for Scikit-learn contains drop-in replacement functionality for the stock scikit-learn package. You can take advantage of the performance optimizations of Intel Extension for Scikit-learn by adding just two lines of code before the usual scikit-learn imports. Intel® Extension for Scikit-learn patching affects performance of specific Scikit-learn functionality.
# 
# ### Data: covtype
# 
# We will use forest cover type dataset known as covtype and fetch the data from sklearn.datasets
# 
# 
# Here we are **predicting forest cover type** from cartographic variables only (no remotely sensed data). The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables (wilderness areas and soil types).
# 
# This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.
# 
# 
# Predicting forest cover type from cartographic variables only (no remotely sensed data). The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables (wilderness areas and soil types).
# 
# This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

# Copyright 2023 Intel Corporation
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
import seaborn as sns
import time

from  sklearn.datasets import fetch_covtype
x, y = fetch_covtype(return_X_y=True)

# for sake of time is 1/4th of the data
subset = x.shape[0]//2
x = x[:subset,:]
y = y[:subset]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=72)

params = {
    'n_neighbors': 40,  
    'weights': 'distance',  
    'n_jobs': -1
}
print('dataset shape: ', x.shape)


################# Insert Patch here ####################################
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()
########################################################################


from sklearn.neighbors import KNeighborsClassifier
start_time = time.time()
knn = KNeighborsClassifier(**params).fit(x_train, y_train)
predicted = knn.predict(x_test)
patched_time = time.time() - start_time
print("Time to calculate \033[1m knn.predict in Patched scikit-learn {:4.1f}\033[0m seconds".format(patched_time))

report = metrics.classification_report(y_test, predicted)
print(f"Classification report for kNN:\n{report}\n")
print("time: ", patched_time)


#########################################
#
# Insert unpatch code here
from sklearnex import unpatch_sklearn
unpatch_sklearn()
#
#########################################

# same code used to predict as from above cell - but this one is UNPATCHED
from sklearn.neighbors import KNeighborsClassifier
start_time = time.time()
knn = KNeighborsClassifier(**params).fit(x_train, y_train)
predicted = knn.predict(x_test)
unpatched_time = time.time() - start_time
print("Time to calculate \033[1m knn.predict in UNpatched scikit-learn {:4.1f}\033[0m seconds".format(unpatched_time))

report = metrics.classification_report(y_test, predicted)
print(f"Classification report for kNN:\n{report}\n")

pred_times = [unpatched_time, patched_time]
print("time: ", unpatched_time)

import csv 

with open('data/compareTimes.csv', 'w', newline='') as csvfile: 
    writer = csv.writer(csvfile) 
    writer.writerow([pred_times[0],pred_times[1]])

import csv

with open('data/compareTimes.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    string_times = list(reader)
pred_times = [float(n) for n in string_times[0]]


print('Intel(R) Extensions for scikit-learn* \033[1mKNN acceleration {:4.1f} x!\033[0m'.format( unpatched_time/patched_time))

print("All Done")

