import sys

rootpath = '../../'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *


from load_data import *
dfmus = load_dfmus('ZTFtest5',rootpath=rootpath)

#Loading data with limited columns included
multigal   = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True,rootpath=rootpath,verbose=False)

#multigal.trim_sample()
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
multigal.sigmaRel_sampler(sigma0='free',overwrite=True,zcosmo='zcmb',chromatic=['theta'],common_beta=None)#,chrom_beta=[0.172])
multigal.plot_posterior_samples()






'''#Old
multigal.sigmaRel_sampler(sigma0='free',overwrite=False,zcosmo='zcmb',chromatic=['theta','etaAV'],common_beta=None,chrom_beta=[-1.2,-0.3])#chrom_beta=[0.17])
multigal.plot_posterior_samples()
multigal.sigmaRel_sampler(sigma0='free',overwrite=False,zcosmo='zcmb',chromatic=['theta','etaAV'],common_beta=True,chrom_beta=[0.14,-0.26])#chrom_beta=[0.14])
multigal.plot_posterior_samples()
multigal.sigmaRel_sampler(sigma0='free',overwrite=False,zcosmo='zcmb',chromatic=['theta','etaAV'],common_beta=False,chrom_beta=[0.15,-0.25,0.4,-0.16])#chrom_beta=[0.14,-0.2])
multigal.plot_posterior_samples()
#'''
