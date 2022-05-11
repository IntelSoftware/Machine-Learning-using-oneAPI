
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

# run like this:
#    python -m sklearnex ./sklearn_sycl.py

import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN

from sklearn.datasets import load_iris
import dpctl

def k_means_random(gpu_device):
    import time
    print("KMeans_random")
    X = np.array([[1., 1.], [1., 2.], [2., 2.], [2., 3.],
                 [1.5, 1.5], [1.7, 2.], [2.8, 1.7], [1.8, 2.7],
                 [10.1, 10.1], [10.2, 12.2], [12.1, 12.1], [12.2, 13.2],
                 [10.8, 10.3], [10.9, 12.9], [11.1, 11.7], [11.2, 12.2],
                 [5.3, 5.], [6.1, 5.2], [5.1, 6.3], [5., 5.1]], dtype=np.float32)
    
    x_device = dpctl.tensor.from_numpy(X, usm_type = 'device', 
                                       queue=dpctl.SyclQueue(gpu_device))
    start = time.time()
    kmeans = KMeans(n_clusters=3, random_state=0, init='random').fit(x_device)
    end = time.time()
    print("time: {}s".format(end-start))
    # save the model and data computed on node w GPU
    # for later condumption on a different node for plotting
    from joblib import dump, load
    dump(kmeans, 'data/kmeans.joblib') 
    dump(X, 'data/X.joblib')     
    print("kmeans.labels_")
    print("X.shape", X.shape)
    print(kmeans.labels_)
    print("kmeans.predict([[0, 0], [12, 3]])")
    print(kmeans.predict(np.array([[0, 0], [12, 3]], dtype=np.float32)))
    print("kmeans.cluster_centers_")
    print(kmeans.cluster_centers_)
    return kmeans, end-start

def logistic_regression_lbfgs(gpu_device):
    import time
    print("LogisticRegression solver='lbfgs'")
    X, y = load_iris(return_X_y=True)
          
    x_device = dpctl.tensor.from_numpy(X, usm_type = 'device', 
                                       queue=dpctl.SyclQueue(gpu_device))
    y_device = dpctl.tensor.from_numpy(y, usm_type = 'device', 
                                       queue=dpctl.SyclQueue(gpu_device))

    start = time.time()
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(
        x_device,
        y_device)
    end = time.time()   
    from joblib import dump, load
    dump(clf, 'data/LogisticRegression.joblib')     
    print("clf.predict(X[:2, :])")
    print(clf.predict(X[:2, :]))
    print("clf.predict_proba(X[:2, :])")
    print(clf.predict_proba(X[:2, :]))
    print("clf.score(X, y)")
    print(clf.score(X, y))
    return clf, end-start

if __name__ == "__main__":
    examples = [
        k_means_random,
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
        device = gpu_device
        print("GPU targeted: ", device)
    else:
        device = cpu_device
        print("CPU targeted: ", device)
        
    for e in examples:
        print("*" * 80)
        try:
            model, elapsed = e(device)
        except:
            print(e, " Failed")
        print("*" * 80)

    print('All looks good!')
