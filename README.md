## Machine Learning using oneAPI
Machine Learning using oneAPI

## Preparation to run on Intel Developer Cloud
To Access via Intel(R) Tiber(tm) Developer Cloud:

1) Go to the Intel Tiber Developer Cloud by visiting
https://cloud.intel.com/
2) Click "Get Started"
3) Subscribe to the “Standard – Free” service tier and complete your cloud
registration.
4) To start up a free and quick Jupyter notebook session with the latest 4th Gen
Intel Xeon CPU and Intel Data Center GPU Max 1100, click the "Training and
Workshops" icon and then "Launch JupyterLab", or one of the specific training
materials launches.

From a Jupyterhub terminal instance:
- mkdir MLoneAPI
- cd MLoneAPI
- git clone https://github.com/IntelSoftware/Machine-Learning-using-oneAPI.git
- cd  Machine-Learning-using-oneAPI
- . lab_setup.sh
- pip install -r requirements.txt

## Currently Known Issues:

There are issues installing the dpctl library from either conda or pip. This impacts the GPU exercises for scikit-learn. You will have to pass on these exercises until the issue is resolved. We will update this README when it is confirmed the problem has been fixed

### Known issue: 

## Purpose
The Jupyter Notebooks in this training are intended to give instructors an accesible but challenging introduction to machine learning using oneAPI.  It enumerates and describes many commonly used Scikit-learn* allgorithms which are used  daily to address machine learning challenges.  The primary purpose is to accelerate commonly used Scikit-learn algorithms for Intel CPUs and GPU's using Intel Extensions for Scikit-learn* which is part of the Intel AI Tools powered by oneAPI.

This workshop is designed to be used on the Intel Developer Cloud.

## License  
Code samples 
are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.
Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Content Details

#### Pre-requisites

- Python* Programming
- Calculus
- Linear algebra
- Statistics


#### Content Structure

Each module folder has a Jupyter Notebook file (`*.ipynb`), this can be opened in Jupyter Lab to view the training contant, edit code and compile/run. 

## Install Directions

The training content can be accessed locally on the computer after installing necessary tools, or you can directly access using Intel Developer Cloud without any installation.

#### Local Installation of JupyterLab and oneAPI Tools

The Jupyter Notebooks can be downloaded locally to computer and accessed:
- Install Jupyter Lab on local computer: [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- Install Intel oneAPI Base Toolkit on local computer: [Installation Guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html)
- conda install -c intel -c conda-forge --override-channels xgboost scikit-learn-intelex modin-all=0.26.1 python=3.10
- git clone the repo and access the Notebooks using Jupyter Lab


#### Access using Intel Developer Cloud 

The Jupyter notebooks are tested and can be run on Intel Developer Cloud  without any installation necessary, below are the steps to access these Jupyter notebooks on Intel DevCloud:
1. Register on [Intel Developer Cloud](https://cloud.intel.com)
2. Login, Get Started and Launch Jupyter Lab
3. Open Terminal in Jupyter Lab and git clone the repo and access the Notebooks
