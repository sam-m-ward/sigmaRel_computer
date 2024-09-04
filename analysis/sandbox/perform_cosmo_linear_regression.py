import sys

rootpath = '../../'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *


from load_data import *
dfmus = load_dfmus('ZTFtest5',rootpath=rootpath)

#Loading data with limited columns included
multigal   = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True,rootpath=rootpath,verbose=False)

multigal.trim_sample()
'''
multigal.resmodel.plot_res(px='theta',annotate_mode='legend')
multigal.resmodel.plot_res(px='AV')
multigal.resmodel.plot_res(px='etaAV')
multigal.resmodel.plot_res(px='theta',py='etaAV')

#multigal.trim_sample()
#multigal.restore_sample()
multigal.resmodel.sample_posterior(pxs=['theta'],overwrite=True)
multigal.resmodel.plot_posterior_samples()
err=1/0
#'''

multigal.n_warmup   = 5000
multigal.n_sampling = 10000
multigal.sigmaRel_sampler(sigma0='free',overwrite=True,zcosmo='zcmb',chromatic=['theta','etaAV'],chrom_beta=[0,0])
multigal.plot_posterior_samples()
