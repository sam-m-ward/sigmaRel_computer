# Instructions for sigmaRel_computer installation

### 1) Clone the repo 

Create a new directory, and in that, clone the GitHub repo

`git clone git@github.com:sam-m-ward/sigmaRel_computer.git`

### 2) Create a new `(sigmaRel)` conda environment, built on cmdstanpy version 1.0.0

`conda create -n sigmaRel -c conda-forge cmdstanpy==1.0.0` <br>
`conda activate sigmaRel` <br>
`pip install astropy arviz jupyter`

### All done!
