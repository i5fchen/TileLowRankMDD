
## Overview
This library is a project that focuses on solving multi-dimensional deconvolution problems using tile low-rank factorization for regularization. It can also be used for other types of inverse problems where the unknown variable shows local low-rank characteristics.

## Project structure

This repository is organized as follows:

* :open_file_folder: ****tilelowrankmdd****: python library containing routines to perform tile low-rank factorization and the solver for inverse problems;

* :open_file_folder: ****data****: where the input data should locate to run experiments under ****example/notebooks****; All related data to reproduce the results are available on [https://zenodo.org/records/11207932](https://zenodo.org/records/11207932)

* :open_file_folder: ****example/notebooks****: set of jupyter notebooks used to test the solver with small scale MDD problems;

* :open_file_folder: ****example/scripts****: set of scripts used to solve large-scale MDD problems in distributed mode;

## Getting started :space_invader: :robot:

To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```
Remember to always activate the environment by typing:
```
conda activate tilelowrank
```
## Citation
If you find this library useful in your research, please consider citing:
```BibTeX
@article{mdd_global_lowrank_geo2024,
author = { Fuqiang Chen  and  Matteo Ravasi  and  David Keyes },
title = {A reciprocity-aware, low-rank regularization for multidimensional deconvolution},
journal = {GEOPHYSICS},
volume = {0},
number = {ja},
pages = {1-65},
year = {2024},
doi = {10.1190/geo2023-0749.1}
}
```


