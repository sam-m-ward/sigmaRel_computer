# sigmaRel_computer
Repo for Hierarchically Analysing Siblings Distance Estimates
---
![Logo](logo/SigmaRelCartoon.png)
---
sigmaRel_computer is a modular Python+Stan pipeline for hierarchical Bayesian analysis of supernova siblings (supernovae that exploded in the same galaxy). This repo takes in individual distance estimates to supernova siblings, and computes posteriors of the relative intrinsic scatter, $\sigma_{\rm{Rel}}$. This is the intrinsic scatter of siblings distance estimates relative to one another within a galaxy. It quantifies the contribution towards the total intrinsic scatter, $\sigma_0$, from within-galaxy variations about the siblings' common properties. Therefore, the contrast of $\sigma_{\rm{Rel}}$ with $\sigma_0$ indicates whether it is within-galaxy variations ($\sigma_{\rm{Rel}}\approx\sigma_0$), or the population variation of the siblings' common properties ($\sigma_{\rm{Rel}} \ll \sigma_0$) that contributes most towards the systematic error in the supernova distance estimates. 

## Analysis variants
This pipeline can be used to perform a single-galaxy siblings analysis, and a multi-galaxy siblings analysis.

### Single-galaxy Analysis
For a single-galaxy analysis, sigmaRel_computer uses the `siblings_galaxy` class, which can be used to:
  1) Visualise individual distance estimates
  2) Combine them by computing a common-distance posterior, by marginalising over $\sigma_{\rm{Rel}}$ with an informative prior: $\sigma_{\rm{Rel}} \sim U(0,\sigma_0)$
  3) Compute analytic posteriors on $\sigma_{\rm{Rel}}$ for different choices of prior upper bound.

### Multi-galaxy Analysis
To perform a multi-galaxy siblings analysis, sigmaRel_computer uses the `multi_galaxy` class, which can be used to:
  1) Compute a cosmology-independent $\sigma_{\rm{Rel}}$ posterior, and plot the samples
  2) Additionally fit for $\sigma_0$, using external distance constraints
  3) Additionally fit for $\sigma_{\rm{pec}}$, also using external distance constraints

There is a high degree of user freedom in the multi-galaxy analysis, where $\sigma_0$, $\sigma_{\rm{pec}}$ and/or the ratio $\sigma_{\rm{Rel}} / \sigma_0$ can all be individually fitted for, or frozen.

## Getting Started

Check out [**installation_instructions.md**](https://github.com/sam-m-ward/sigmaRel_computer/blob/main/installation_instructions.md) to clone the repo and set up a conda environment.

See [**tutorial.ipynb**](https://github.com/sam-m-ward/birdsnack/blob/main/demo_notebook.ipynb) for a quick practitioner's introduction to sigmaRel_computer.

## Acknowledgements

sigmaRel_computer was developed by Sam M. Ward. 

The relative intrinsic scatter formalism for hierarchically analysing siblings was developed in [**Ward et al. 2023**](https://ui.adsabs.harvard.edu/abs/2022arXiv220910558W/abstract). Please cite when using this code.
