import sys

rootpath = './'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *
from simulate_distances import *

simulator = SiblingsDistanceSimulator(Ng=100,Sg=3,external_distances=True)
dfmus     = simulator.dfmus

multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True)
#multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[0.1,0.15,1.0])
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True)
multigal.plot_posterior_samples()
