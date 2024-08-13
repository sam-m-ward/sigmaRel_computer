import sys
rootpath = '../../'
sys.path.append(rootpath+'model_files/')

from load_data import *
from ResModel import *
from sigmaRel import *


dfmus    = load_dfmus('ZTFtest5',rootpath=rootpath)
multigal = multi_galaxy_siblings(dfmus,rootpath=rootpath)
#multigal.trim_sample()
#multigal.plot_parameters(PAR=['mu','AV','theta'])#,subtract_g_mean=True)


rml = ResModelLoader(multigal.dfmus,rootpath=rootpath)

rml.plot_res(px='theta')
rml.plot_res(px='etaAV')
rml.plot_res(px='theta',py='etaAV')
#rml.plot_res(px='AV')
#rml.plot_res(px='theta',py='AV')
'''#Plots
rml.plot_res_pairs(px='theta')
rml.plot_res_pairs(px='AV')
rml.plot_res_pairs(px='theta',py='AV')
rml.plot_res(px='theta',py='AV')
rml.plot_res(px='theta',py='etaAV')
rml.plot_res(px='theta',py='AV',zerox=False,zeroy=False)
rml.plot_res(px='theta',py='AV',zerox=False)
rml.plot_res(px='theta',py='AV',zeroy=False)
rml.plot_res(px='theta',zerox=False)
rml.plot_res(px='AV',zerox=False)
#'''
#err=1/0
#Need lots of samples for cosmo-indep mu-etaAV on cosmo subsample
#rml.n_warmup   = 20000
#rml.n_sampling = 10000
#etaAVs end up with large beta values only at high rho. This makes sense. If rho is 1, there is no within-galaxy dispersion of AV, in which case
#it's all in the common component. Therefore, large beta doesn't translate to large dmu discrepancy, all ends up in common mu, which is unbounded in
#cosmo-indep analysis. The reason rho=1 is permitted is because of large uncertainties in etaAV.

#rml.sample_posterior(py='etaAV',beta=[0],overwrite=True)#,beta=[0.102])#.14])
rml.sample_posterior(pxs=['theta'],overwrite=True,alpha=False)#,sigint_prior=[1e-6,1e-6],sigint_const=[100,1])
#rml.sample_posterior(pxs=['theta','etaAV'],overwrite=True,beta=[0.13,-0.03])#,beta=[0.102])#.14])
#rml.plot_posterior_samples()
#rml.sample_posterior(pxs=['theta'],overwrite=True)#,beta=[0.102])#.14])
#rml.plot_posterior_samples()
#rml.sample_posterior(pxs=['theta'],overwrite=True,beta=[0.143])#,beta=[0.102])#.14])
#rml.plot_posterior_samples()
#rml.sample_posterior(pxs=['theta'],overwrite=True,beta=[0])#,beta=[0.102])#.14])
rml.plot_posterior_samples()
