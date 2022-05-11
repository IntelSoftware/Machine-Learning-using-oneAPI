from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection as model_selection
from sklearnex import patch_sklearn, unpatch_sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import datasets
import pandas as pd
import numpy as np
import dpctl
import time

connect4 = pd.read_csv('data/connect-4.data')

data = connect4.iloc[:,:42].replace(['x', 'o', 'b'], [0,1,2])

keep = .25
subsetLen = int(keep*data.shape[0])

X = np.byte( data.iloc[:subsetLen,:].to_numpy() )

X = X[:subsetLen,:42]
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
enc.categories_

XOHE = np.short(enc.transform(X).toarray() )# X one hot encoded

#Data_y = connect4.iloc[:,42].to_numpy() #non-numeric values
Data_y = connect4.iloc[:,42].\
    replace(['win', 'loss', 'draw'], [0, 1, 2]).\
    to_numpy()  # convert non_numeric to numeric

y =  Data_y[:subsetLen] 

X_train, X_test, y_train, y_test = model_selection.train_test_split(XOHE, y, train_size=0.80, test_size=0.20, random_state=101)

params = {
    'n_neighbors': 40,
    'weights': 'distance',
    'n_jobs': -1
}

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

if not gpu_available:
    print("GPU Not Available")
else:
    patch_sklearn()
    #from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report

    x_device = dpctl.tensor.from_numpy(X_train, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))
    y_device = dpctl.tensor.from_numpy(y_train, usm_type = 'device', queue=dpctl.SyclQueue(gpu_device))

    start_time = time.time()
    knn = KNeighborsClassifier(**params).fit(x_device, y_device)
    predicted = knn.predict(X_test)
    end_time = time.time()
    print("Time to calculate knn.predict in stock scikit-learn {:4.1f} seconds".format(end_time - start_time))
    report = classification_report(y_test, predicted)
    print(f"Classification report for kNN:\n{report}\n")
