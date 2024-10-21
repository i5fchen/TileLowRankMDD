#!/bin/bash
# 
# Installer for tilelowrank_mdd environment
#
# Run: ./install_env.sh
#
# FC, 21/10/2024

echo 'Creating tilelowrank environment'

# create conda env
conda env create --file environment.yml --debug 
source ~/miniconda3/etc/profile.d/conda.sh
# source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate tilelowrank
echo 'Created and activated environment:' $(which python)

echo 'Done!'
