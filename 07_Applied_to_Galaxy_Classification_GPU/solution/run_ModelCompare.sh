#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling Machine Learning using oneAPI -- Practicum_analyzeGalaxyBatch.py
python Practicum_analyzeGalaxyBatch.py

