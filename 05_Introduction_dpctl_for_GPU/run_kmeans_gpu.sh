#!/bin/bash
# source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
source  /glob/development-tools/versions/oneapi/2022.3.1/inteloneapi/setvars.sh --force > /dev/null 2>&1
conda activate base
python lab/scikit_kmeans.py

