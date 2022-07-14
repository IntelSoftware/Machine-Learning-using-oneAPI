
# Copyright 2021 Intel Corporation
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

import numpy as np

################################################################
#
# Import dptcl and patch here

import dpctl
print("dpctl.__version__ = ", dpctl.__version__)

from sklearnex import patch_sklearn
patch_sklearn()
#
################################################################

X = np.array([[1,  2], [1,  4], [1,  0],
              [10, 2], [10, 4], [10, 0]])

# You need to re-import scikit-learn algorithms after the patch
from sklearn.cluster import KMeans

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
    x_device = dpctl.tensor.from_numpy(X, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))
    kmeans = KMeans(n_clusters=2, init='random', random_state=0).fit(x_device)
    print(f"kmeans.labels_ = {kmeans.labels_}")
else:
    print("GPU not found")
