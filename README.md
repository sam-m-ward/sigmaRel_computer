# sigmaRel_computer
Repo for Hierarchically Analysing Siblings Distance Estimates
---
![Logo](logo/SigmaRelCartoon.png)
---
sigmaRel_computer is a modular Python+Stan pipeline for hierarchical Bayesian analyses of supernova siblings (supernovae that exploded in the same galaxy). Using photometric distance estimates to individual SN siblings, sigmaRel_computer can be used to perform a:
* ***Cosmology-independent analysis***: Compute a cosmology-independent posterior on the relative intrinsic scatter, $\sigma_{\rm{Rel}}$.

Additionally using redshift-based cosmology distances, sigmaRel_computer can be used to perform a:
* ***Cosmology-dependent analysis***: Compute a joint posterior on $\sigma_{\rm{Rel}}$ and the total intrinsic scatter, $\sigma_0$. 

The relative scatter, $\sigma_{\rm{Rel}}$, is the residual scatter of individual siblings distance estimates relative to one another within a galaxy. It quantifies the contribution towards the total intrinsic scatter, $\sigma_0$, from within-galaxy variations about the siblings' common properties in each galaxy. Therefore, the contrast of $\sigma_{\rm{Rel}}$ with $\sigma_0$ indicates whether it is within-galaxy variations ($\sigma_{\rm{Rel}}\approx\sigma_0$), or the population variation of the siblings' common properties ($\sigma_{\rm{Rel}} \ll \sigma_0$) that contributes most towards the total intrinsic scatter in the Hubble diagram. 

### Multi-galaxy Analysis
The default analysis pipeline is the `multi_galaxy_siblings` class, used for:
  1) ***Cosmology-independent analyses***, including computing/plotting posteriors for different choices of $\sigma_{\rm{Rel}}$ hyperprior. 
  2) ***Cosmology-dependent analyses***, including computing/plotting posteriors for different assumptions about e.g. the intrinsic scatter hyperpriors, $\sigma_{\rm{pec}}$, and whether to model latent distance or redshift parameters.
  3) Visualising individual photometric distances estimates, and Hubble diagrams.
     
### Single-galaxy Analysis
This repo also includes the `siblings_galaxy` class for analysing a single siblings-galaxy, used for:
  1) ***Cosmology-independent analyses***, including computing/plotting posteriors for different choices of $\sigma_{\rm{Rel}}$ hyperprior.
  2) Computing a common-distance posterior by marginalising over $\sigma_{\rm{Rel}}$ with an informative hyperprior.
  3) Visualising individual photometric distance estimates.

## Getting Started

Check out [**installation_instructions.md**](https://github.com/sam-m-ward/sigmaRel_computer/blob/main/installation_instructions.md) to clone the repo and set up a conda environment.

See [**tutorial.ipynb**](https://github.com/sam-m-ward/sigmaRel_computer/blob/main/tutorial.ipynb) for a quick introduction to sigmaRel_computer.

## Example Use

```python
#Set path to model_files
import sys
rootpath = './'
sys.path.append(rootpath+'model_files/')

#Import sigmaRel_computer
from sigmaRel import *

#Synthetic data: 15 sibling-pair galaxies; sigma0=0.1; sigmaRel=0.01
dfmus = {
'SN': ['S1_G1', 'S2_G1', 'S1_G2', 'S2_G2', 'S1_G3', 'S2_G3', 'S1_G4', 'S2_G4', 'S1_G5', 'S2_G5', 'S1_G6', 'S2_G6', 'S1_G7', 'S2_G7', 'S1_G8', 'S2_G8', 'S1_G9', 'S2_G9', 'S1_G10', 'S2_G10', 'S1_G11', 'S2_G11', 'S1_G12', 'S2_G12', 'S1_G13', 'S2_G13', 'S1_G14', 'S2_G14', 'S1_G15', 'S2_G15'], 'Galaxy': ['G1', 'G1', 'G2', 'G2', 'G3', 'G3', 'G4', 'G4', 'G5', 'G5', 'G6', 'G6', 'G7', 'G7', 'G8', 'G8', 'G9', 'G9', 'G10', 'G10', 'G11', 'G11', 'G12', 'G12', 'G13', 'G13', 'G14', 'G14', 'G15', 'G15'],
'mus': [35.106976115957465, 35.03526597340868, 36.83107915740559, 36.893906053332145, 35.2557424918456, 35.27092132371533, 34.40281973787297, 34.36608598937359, 37.129838540516516, 37.04959329387558, 36.11926041073685, 36.07068367746133, 37.59776498381559, 37.557724421424005, 36.20141010582293, 36.25286074492239, 37.67391556059631, 37.71006833495005, 36.93762508216655, 36.86242928977509, 37.695246044925305, 37.6851755445182, 36.63180632432816, 36.618709339412874, 37.85050920083928, 37.90525141513808, 36.09793302296473, 36.10607890014568, 35.652157744703736, 35.676332232572356],
'mu_errs': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
'zhelio_hats': [0.025312026871142777, 0.025312026871142777, 0.05578908991913875, 0.05578908991913875, 0.027784922713810972, 0.027784922713810972, 0.01829057323455145, 0.01829057323455145, 0.057569954799814214, 0.057569954799814214, 0.04239258317190128, 0.04239258317190128, 0.08019072806161756, 0.08019072806161756, 0.03994542882436594, 0.03994542882436594, 0.08067300293135135, 0.08067300293135135, 0.055588532681127296, 0.055588532681127296, 0.0826060076991514, 0.0826060076991514, 0.04868689976235743, 0.04868689976235743, 0.09419306106244194, 0.09419306106244194, 0.037131737714328185, 0.037131737714328185, 0.03201362178571704, 0.03201362178571704],
'zhelio_errs': [1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06],
'zHD_hats': [0.025312026871142777, 0.025312026871142777, 0.05578908991913875, 0.05578908991913875, 0.027784922713810972, 0.027784922713810972, 0.01829057323455145, 0.01829057323455145, 0.057569954799814214, 0.057569954799814214, 0.04239258317190128, 0.04239258317190128, 0.08019072806161756, 0.08019072806161756, 0.03994542882436594, 0.03994542882436594, 0.08067300293135135, 0.08067300293135135, 0.055588532681127296, 0.055588532681127296, 0.0826060076991514, 0.0826060076991514, 0.04868689976235743, 0.04868689976235743, 0.09419306106244194, 0.09419306106244194, 0.037131737714328185, 0.037131737714328185, 0.03201362178571704, 0.03201362178571704],
'zHD_errs': [1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06],
'cosmo_sample': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
}

#Load in data and initialise for cosmo-dep. analysis using fixed sigma_pec=250km/s.
multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True,rootpath=rootpath)
#Perform cosmo-dep. fit
multigal.sigmaRel_sampler()
#Plot posterior samples
multigal.plot_posterior_samples()
```

## Acknowledgements

sigmaRel_computer was developed by Sam M. Ward. 

The relative scatter forward model for hierarchical siblings analyses was developed in [**Ward et al. 2023**](https://ui.adsabs.harvard.edu/abs/2022arXiv220910558W/abstract); please cite when using this code.

