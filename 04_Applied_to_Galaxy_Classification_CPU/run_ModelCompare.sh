#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI Essentials Module1 -- DPPY kmeans gpu - 1 of 2 04_analyzeGalaxyBatch.py
python Practicum_analyzeGalaxyBatch.py

