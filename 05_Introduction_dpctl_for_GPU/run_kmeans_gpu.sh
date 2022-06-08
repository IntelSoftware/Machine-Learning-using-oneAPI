#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI Essentials Module1 -- scikit-learn-Intelex_Kmeans - 2 of 3 kmeans_fit_then_predict_gpu.py
python lab/scikit_kmeans.py

