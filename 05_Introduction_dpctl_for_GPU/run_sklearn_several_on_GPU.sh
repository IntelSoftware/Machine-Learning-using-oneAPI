#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI Essentials Module1 -- scikit-learn-Intelex_Intro - dpctl_several_functions.py
python lab/dpctl_several_functions.py

