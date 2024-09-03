import sys

rootpath = '../../'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *


from load_data import *
dfmus = load_dfmus('ZTFtest5',rootpath=rootpath)

#Loading data with limited columns included
multigal   = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True,rootpath=rootpath,verbose=False)

#multigal.trim_sample()
multigal.resmodel.plot_res(px='theta',annotate_mode='legend')
multigal.resmodel.plot_res(px='AV')
multigal.resmodel.plot_res(px='etaAV')
multigal.resmodel.plot_res(px='theta',py='etaAV')

#multigal.trim_sample()
#multigal.restore_sample()
multigal.resmodel.sample_posterior(pxs=['theta'],overwrite=False)
multigal.resmodel.plot_posterior_samples()
