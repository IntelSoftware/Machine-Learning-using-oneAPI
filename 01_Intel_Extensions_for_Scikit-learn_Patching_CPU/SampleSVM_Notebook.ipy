#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# # Sample Notebook: 
# 
# ### Use nbconvert  patch_sklearn from command line
# 
# * On your DevCloud instance, click the blue + in the upper left of browser (the launcher)
# * Scroll down in the launcher and Launch a Terminal
# * In the terminal:
# 1) cd ai_learning_paths/Essentials\ of\ ML\ using\ AI\ Kit/01_sklernex_Intro/ 
# 1) jupyter nbconvert --to script SampleSVM_Notebook.ipynb
# 1) python -m sklearnex SampleSVM_Notebook.py
# 
# The above should run the python script and apply the sklearnex patch to the entire python file prior to executing the file

# In[20]:


from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import time
import numpy as np
import numpy.ma as ma

print("\nThere will be two runs of this code if you patched correctly.")
print("One run that runs faster which is patched (runs in about 11 seconds on DevCloud) ,")
print("followed by one which runs slowly which is unpatched (run for several minutes on DevCloud)\n")

connect4 = pd.read_csv('data/connect-4.data')

data = connect4.iloc[:,:42].replace(['x', 'o', 'b'], [0,1,2])

keep = .25
subsetLen = int(keep*data.shape[0])

X = np.byte( data.iloc[:subsetLen,:].to_numpy() )
#np.random.seed(42)
#np.random.shuffle(X)
X = X[:subsetLen,:42]
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
enc.categories_

XOHE = np.short(enc.transform(X).toarray() )# X one hot encoded

Data_y = connect4.iloc[:,42].to_numpy()
#np.random.shuffle(Data_y)
y =  Data_y[:subsetLen] 

X_train, X_test, y_train, y_test = model_selection.train_test_split(XOHE, y, train_size=0.80, test_size=0.20, random_state=101)


# In[22]:


y.shape


# In[21]:


from sklearnex import patch_sklearn, unpatch_sklearn
#patch_sklearn()
from sklearn.metrics import classification_report

def predict( linear ):
    import numpy as np
    time_patch_predict = time.time()
    y_pred = linear.predict(X_test)
    elapsed = time.time() - time_patch_predict
    return elapsed, y_pred

def fit():
    start = time.time()
    linear = svm.SVC(kernel='linear', C=100).fit(X_train, y_train)
    time_patch_fit =  time.time() - start
    return time_patch_fit, linear


# In[23]:


from sklearn.metrics import classification_report
#patch_sklearn()
from sklearn import svm
time_fit, linear = fit()
time_predict, y_pred = predict(linear)
target_names = ['win', 'loss', 'draw']
print("file as is ")
print(classification_report(y_test, y_pred, target_names=target_names))
print('Elapsed time: {:.2f} sec'.format( time_fit + time_predict))


# In[1]:


from sklearnex import patch_sklearn, unpatch_sklearn
from sklearn.metrics import classification_report
unpatch_sklearn("svc")
from sklearn import svm
time_fit, linear = fit()
time_predict, y_pred = predict(linear)
target_names = ['win', 'loss', 'draw']
print("explicit unpatch ")
print(classification_report(y_test, y_pred, target_names=target_names))
print('Elapsed time: {:.2f} sec'.format( time_fit + time_predict))


# In[ ]:





# In[ ]:




