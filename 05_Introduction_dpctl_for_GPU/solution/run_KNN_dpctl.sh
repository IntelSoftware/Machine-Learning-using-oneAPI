#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI Essentials Module1 -- scikit-learn-Intelex_Intro - 4 of 5 KNN_dpctl.py
python lab/compute_KNN_GPU.py


