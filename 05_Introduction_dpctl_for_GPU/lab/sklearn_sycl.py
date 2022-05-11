
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

# daal4py Scikit-Learn examples for GPU
# run like this:
#    python -m sklearnex ./sklearn_sycl.py

import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN

from sklearn.datasets import load_iris
import dpctl

def k_means_random(gpu_device):
    print("KMeans init='random'")
    X = np.array([[1., 2.], [1., 4.], [1., 0.],
                  [10., 2.], [10., 4.], [10., 0.]], dtype=np.float32)
    
    x_device = dpctl.tensor.from_numpy(X, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))
    kmeans = KMeans(n_clusters=2, random_state=0, init='random').fit(x_device)
    #kmeans = KMeans(n_clusters=2).fit(x_device)

    print("kmeans.labels_")
    print(kmeans.labels_)
    print("kmeans.predict([[0, 0], [12, 3]])")
    print(kmeans.predict(np.array([[0, 0], [12, 3]], dtype=np.float32)))
    print("kmeans.cluster_centers_")
    print(kmeans.cluster_centers_)


def linear_regression(gpu_device):
    print("LinearRegression")
    X = np.array([[1., 1.], [1., 2.], [2., 2.], [2., 3.]], dtype=np.float32)
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2], dtype=np.float32)) + 3
           
    x_device = dpctl.tensor.from_numpy(X, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))
    y_device = dpctl.tensor.from_numpy(y, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))

    reg = LinearRegression().fit(x_device, y_device)
    print("reg.score(X, y)")
    print(reg.score(X, y))
    print("reg.coef_")
    print(reg.coef_)
    print("reg.intercept_")
    print(reg.intercept_)
    print("reg.predict(np.array([[3, 5]], dtype=np.float32))")
    print(reg.predict(np.array([[3, 5]], dtype=np.float32)))


def logistic_regression_lbfgs(gpu_device):
    print("LogisticRegression solver='lbfgs'")
    X, y = load_iris(return_X_y=True)
          
    x_device = dpctl.tensor.from_numpy(X, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))
    y_device = dpctl.tensor.from_numpy(y, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))

    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(
        x_device,
        y_device)
    print("clf.predict(X[:2, :])")
    print(clf.predict(X[:2, :]))
    print("clf.predict_proba(X[:2, :])")
    print(clf.predict_proba(X[:2, :]))
    print("clf.score(X, y)")
    print(clf.score(X, y))

def dbscan(gpu_device):
    print("DBSCAN")
    X = np.array([[1., 2.], [2., 2.], [2., 3.],
                  [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
    
    x_device = dpctl.tensor.from_numpy(X, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))    
    clustering = DBSCAN(eps=3, min_samples=2).fit(x_device)
    print("clustering.labels_")
    print(clustering.labels_)
    print("clustering")
    print(clustering)

if __name__ == "__main__":
    examples = [
        k_means_random, 
        dbscan,
        linear_regression,
        logistic_regression_lbfgs,
    ]

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
        
    device = gpu_device
    for e in examples:
        print("*" * 80)
        try:
            e(device)
        except:
            print(e, " Failed")
        print("*" * 80)

    print('All looks good!')
