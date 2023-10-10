#!/bin/bash

error_exit() {
    echo "$1" 1>&2
    exit 1
}

setup_oneapi() {
    source /opt/intel/oneapi/setvars.sh --force || error_exit "Failed to set up Intel OneAPI."
}

create_conda_environment() {
    ENV_NAME=${1:-MLoneAPI}
    conda config --add channels intel || error_exit "Failed to add Intel channel."
    conda create -y -n "$ENV_NAME" intelpython3_full || error_exit "Failed to create Conda environment."
    source $(conda info --base)/etc/profile.d/conda.sh || error_exit "Failed to source Conda profile."
    conda activate "$ENV_NAME" || error_exit "Failed to activate Conda environment."
}

install_packages() {
    conda install -y -c conda-forge jupyterlab ipywidgets numpy scipy || error_exit "Failed to install Jupyter and widgets."
    conda install -y -c anaconda pip seaborn pandas || error_exit "Failed to install Seaborn and Pandas."
    python -m pip install Pillow opencv-python plotly tqdm matplotlib || error_exit "Failed to install additional packages."
}

prepare_data() {
    echo "No data copy is required"
}

download_models() {
    echo "No models are required for download"

}

main() {
    prepare_env=true
    prepare_data=true
    download_models=true
    for arg in "$@"; do
        case $arg in
            --prepare_env) prepare_env=true ;;
            --prepare_data) prepare_data=true ;;
            --download_models) download_models=true ;;
            *) error_exit "Unknown argument: $arg" ;;
        esac
    done
    if [ "$prepare_env" = true ]; then
        setup_oneapi
        create_conda_environment "MLoneAPI"
        install_packages
    fi
    if [ "$prepare_data" = true ]; then
        prepare_data
    fi
    if [ "$download_models" = true ]; then
        download_models
    fi
    echo "Script execution successful!"
}

main "$@"
