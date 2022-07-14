
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

from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
import dpctl

X = np.array([[1., 2.], [2., 2.], [2., 3.],
            [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)

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
else:
    # target a remote hosy CPU when submitted via q.sh or qsub -I
    x_device = dpctl.tensor.from_numpy(X, usm_type = 'device', queue=dpctl.SyclQueue(cpu_device))

clustering_host = DBSCAN(eps=3, min_samples=2).fit(x_device)

print("DBSCAN.get_params: ",DBSCAN.get_params)
print('data type clustering_host.labels_', type(clustering_host.labels_))

# write meaningful cluster info to a CSV 
clustering_labels = pd.DataFrame(clustering_host.labels_)
clustering_labels.to_csv('data/DBSCAN_Labels.csv', index=False )

clustering_components = pd.DataFrame(clustering_host.components_)
clustering_components.to_csv('data/DBSCAN_Components.csv', index=False )

print("DBSCAN components: ", clustering_host.components_, "\nDBSCAN labels: ",clustering_host.labels_)
