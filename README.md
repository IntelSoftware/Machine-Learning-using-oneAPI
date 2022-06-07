## Title
 Introduction to Machine Learning
  
## Requirements
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 20.04, 20 Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; AI Analytics Tookkit, Jupyter Notebooks, Intel DevCloud
|                                   | pip install -r requirements.txt
  
## Purpose
The Jupyter Notebooks in this training are intended to give instructors an accesible but challenging introduction to machine learning using oneAPI.  It enumerates and describes many commonly used Scikit-learn* allgorithms which are used  daily to address machine learning challenges.  The primary purpose is to accelerate commonly used Scikit-learn algorithms for Intel CPUs and GPU's using Intel Extensions for Scikit-learn* which is part of the Intel AI Analytics Toolkit powered by oneAPI.

This workshop is designed to be used on the DevCloud and includes details on submitting batch jobs on the DevCloud environment.

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


## Syllabus

- 11 Modules (18 hours)
- 11 Lab Exercises

-----------------------
| Folder | Modules | Description | Duration |
| :--- | :--- | :------ | :------ |
| 01_Intel_Extensions_for_Scikit-learn_Patching_CPU |[01_Intel_Extensions_for_Scikit-learn_Patching_CPU](01_Intel_Extensions_for_Scikit-learn_Patching_CPU/01_sklearnex_Intro_Acceleration.ipynb)| + Describe the basics of oneAPI AI Kit components, and where the Intel(R) Extensions for scikit-learn fits in the broader package.<br> + Describe where to download and how to install the oneAPI AI Kit.<br> + Describe the advantages of one specific component of oneAPI AI Kit, Intel(R) Extensions for scikit-learn, invoked via the sklearnex library.<br> + Apply the patch and unpatch functions with varying granularities including python scripts and also within Jupyter cells: from whole file applications to more surgical patches applied to a single algorithm.<br> + Enumerate sklearn algorithms which have been optimized.| 20 min |
| 01_Intel_Extensions_for_Scikit-learn_Patching_CPU |[02_Coarse_Patching_Instructions](01_Intel_Extensions_for_Scikit-learn_Patching_CPU/02_Coarse_Patching_Instructions.ipynb)| + Describe how to import and apply patch_sklearn().<br> + Describe how to import and apply unpatch_sklearn().<br> + Describe method & apply the patch to an entire python program.<br> + Describe how to surgically unpatch specific optimized functions if needed.<br> + Describe a patching strategy that ensures that the Intel Extensions for scikit-learn runs as fast or faster than the stock algorithms it replaces.<br> + Apply patch methodology to speed up KNN on CovType dataset| 20 min |
| 02_Applied_Patching_CPU |[01_Pairwise_DistanceVectorizedStockSImulationReadPortfolio](02_Applied_Patching_CPU/01_Pairwise_DistanceVectorizedStockSImulationReadPortfolio.ipynb)|+ Describe and apply the correct surgical patching method to patch pairwise_distance.<br> + Describe which distance metrics such as 'euclidean', 'mahattan', 'cosine', or 'correlation' are optimized by Intel Extensions for Scikit learn.<br> + Describe the application of pairwise_distance to the problem of finding all time series charts similar to a chosen pattern.| 20 min |
| 02_Applied_Patching_CPU|[02_PatchingKNN_CPU](02_Applied_Patching_CPU/02_PatchingKNN_CPU.ipynb)| + Describe how to surgically unpatch specific optimized functions if needed.<br> + Apply patching to KNN algorithm.<br> + Describe acceleration for the covtype dataset with KNN classification| 20 min |
| 02_Applied_Patching_CPU |[03_Patching_Kmeans_CPU](02_Applied_Patching_CPU/03_Patching_Kmeans_CPU.ipynb)| + Describe the value of Intel® Extension for Scikit-learn methodology in extending scikit-learn optimization capabilites.<br> + Name key imports and function calls to use Intel Extension for Scikit-learn to target Kmeans.<br> + Build a Sklearn implementation of Kmeans targeting CPU using patching.<br> + Apply patching with dynamic versus lexical scope approaches.| 20 min |
| 02_Applied_Patching_CPU |[04_PatchingSVM_CPU](02_Applied_Patching_CPU/04_PatchingSVM_CPU.ipynb)| + Describe how to surgically unpatch specific optimized functions if needed.<br> + Describe differences in patching more globally versus more surgically.<br> + Apply patching to SVC algorithm.<br> + Describe acceleration for the covtype dataset usinf SVC. | 20 min |   
| 03_Applied_to_Image_Clustering_CPU |[01_Practicum_ImageClustering](03_Applied_to_Image_Clustering_CPU/01_Practicum_ImageClustering.ipynb)| + Explore and interpret the image dataset.<br> + Apply Intel® Extension for Scikit-learn* patches to Principal Components Analysis (PCA), Kmeans,and DBSCAN algorithms.<br> + Synthesize your understanding- searching for ways to patch or unpatch any applicable cells to maximize the performance of each cell.| 60 min |
| 04_Applied_to_Galaxy_Classification_CPU |[01_Practicum_AnalyzeGalaxyBatch](04_Applied_to_Galaxy_Classification_CPU/01_Practicum_AnalyzeGalaxyBatch.ipynb)| + Apply Multiple Classification Algorithms with GPU to classify stars belonging to each galaxy within a combined super galaxy to determine most accurate model.<br> + Apply Intel® Extension for Scikit-learn* patch and SYCL context to compute on available GPU resource.<br> Synthesize your compreshension by searching for opportunities in each cell to maximize performance. Investigate adding pairwise distance as a means for all the stars within 3 light years.| 60 min |
| 05_Introduction_dpctl_for_GPU |[01_Introduction_simple_gallery_dpctl_for_GPU](05_Introduction_dpctl_for_GPU/01_Introduction_simple_gallery_dpctl_for_GPU.ipynb)| + Apply patching while targeting an Intel GPU.<br> + Apply Intel Extension for Scikit-learn to KNeighborsClassifier on Intel GPU.| 30 min |
| 05_Introduction_dpctl_for_GPU|[02_PatchingKNN_GPU](05_Introduction_dpctl_for_GPU/02_PatchingKNN_GPU.ipynb)|+ Describe how to apply dpctl compute follows data in conjuction with patching.<br> + Apply patching to KNN algorithm on covtype dataset.| 20 min |
| 05_Introduction_dpctl_for_GPU|[03_Gallery_of_Functions_on_GPU](05_Introduction_dpctl_for_GPU/03_Gallery_of_Functions_on_GPU.ipynb)| + Apply the patch functions with varying granularities.<br> + Leverage the Compute Follows Data methodology using Intel DPCTL library to target Intel GPU.<br> + Apply DPCTL and Patching to variety of Scikit-learn Algorithsm in a simple test harness structure.<br> + For the current hardware configurationson the Intel DevCloud - we are NOT focusing on performance.| 30 min |
| 06_Applied_to_Image_Clustering_GPU|[01_Practicum_ImageClustering](06_Applied_to_Image_Clustering_GPU/01_Practicum_ImageClustering.ipynb)| + Explore and interpret the image dataset.<br> + Apply Intel® Extension for Scikit-learn* patches to Principal Components Analysis (PCA), Kmeans,and DBSCAN algorithms.<br> + Synthesize your understanding- searching for ways to patch or unpatch any applicable cells to maximize the performance of each cell.<br> + Apply a q.sh script to submit a job to another node that has a GPU on Intel DevCloud.| 60 min |  
| 07_Applied_to_Galaxy_Classification_GPU|[01_Practicum_AnalyzeGalaxyBatch](07_Applied_to_Galaxy_Classification_GPU/01_Practicum_AnalyzeGalaxyBatch.ipynb)| + Apply Multiple Classification Algorithms with GPU to classify stars belonging to each galaxy within a combined super galaxy to determine most accurate model.<br> + Apply Intel® Extension for Scikit-learn* patch and SYCL context to compute on available GPU resource.<br> + Synthesize your compreshension by searching for opportunities in each cell to maximize performance. | 60 min |  
| 08_Introduction_to_Numpy_powered_by_oneAPI|[01_Numpy_How_Fast_Are_Numpy_Ops](08_Introduction_to_Numpy_powered_by_oneAPI/01_Numpy_How_Fast_Are_Numpy_Ops.ipynb)| + Desribe why replacing inefficient code, such as time consuming loops, wastes resources, and time.<br> + Describe why using Python for highly repetitive small tasks is inefficient.<br> + Describe the additive value of leveraging packages such as Numpy which are powered by oneAPI in a cloud world.<br> + Describe the importance of keeping oneAPI and 3rd party package such as Numpy, Scipy and others is important.<br> + Enumerate ways in which Numpy accelerates code.<br> + Apply loop replacement methodologies in a variety of scenarios.| 60 min | 
| 08_Introduction_to_Numpy_powered_by_oneAPI |[02_PandasPoweredBy_oneAPI](08_Introduction_to_Numpy_powered_by_oneAPI/02_PandasPoweredBy_oneAPI.ipynb)| + Apply Numpy methods to dramatically speed up certain common Pandas bottlenecks.<br> + Apply WHERE or SELECT in Numpy powered by oneAPI.<br> + Avoid iterrows using Numpy techniques.<br> + Achieve better performacne by converting numerical columns to numpy arrays.| 20 min |  
#### Content Structure

Each module folder has a Jupyter Notebook file (`*.ipynb`), this can be opened in Jupyter Lab to view the training contant, edit code and compile/run. 

## Install Directions

The training content can be accessed locally on the computer after installing necessary tools, or you can directly access using Intel DevCloud without any installation.

#### Local Installation of JupyterLab and oneAPI Tools

The Jupyter Notebooks can be downloaded locally to computer and accessed:
- Install Jupyter Lab on local computer: [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- Install Intel oneAPI Base Toolkit on local computer: [Installation Guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) 
- git clone the repo and access the Notebooks using Jupyter Lab


#### Access using Intel DevCloud

The Jupyter notebooks are tested and can be run on Intel DevCloud without any installation necessary, below are the steps to access these Jupyter notebooks on Intel DevCloud:
1. Register on [Intel DevCloud](https://devcloud.intel.com/oneapi)
2. Login, Get Started and Launch Jupyter Lab
3. Open Terminal in Jupyter Lab and git clone the repo and access the Notebooks
