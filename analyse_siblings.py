import sys

rootpath = './'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *

'''
from simulate_distances import *
simulator = SiblingsDistanceSimulator(Ng=200,Sg=2,external_distances=True,sigmaRel=0.0999,zcmberr=1e-5)
dfmus     = simulator.dfmus
print (dfmus)
#multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec='free',use_external_distances=True)
#multigal = multi_galaxy_siblings(dfmus,sigma0='free',eta_sigmaRel_input=None,sigmapec=250,use_external_distances=True)
multigal = multi_galaxy_siblings(dfmus,sigma0=1.0,eta_sigmaRel_input=None,sigmapec=250,use_external_distances=False)
multigal.compute_analytic_multi_gal_sigmaRel_posterior()
#multigal.sigmaRel_sampler()#sigma0=None, sigmapec=None, eta_sigmaRel_input=None, use_external_distances=None, overwrite=True)
#multigal.plot_posterior_samples()
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,alt_prior=False)
#multigal.plot_posterior_samples()
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,alt_prior=True)
#multigal.plot_posterior_samples()
multigal.sigmaRel_sampler(sigma0='free',sigmapec='free',use_external_distances=True,zmarg=False,overwrite=True,alt_prior=True)
multigal.plot_posterior_samples()
#multigal.sigmaRel_sampler(sigma0='free',sigmapec='free',use_external_distances=True,zmarg=False,overwrite=True,alt_prior=False)
#multigal.plot_posterior_samples()

err=1/0
#'''


from load_data import *
dfmus = load_dfmus('ZTFtest5')

#Inflate error for 20abmarcv_ siblings
#dfmus[dfmus['Galaxy']==1]['zcmb_errs'] = 0.01

#APPLY FILTERING
#dfmus = dfmus[~dfmus['Galaxy'].isin([1,3,6,11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,11,12])]
dfmus = dfmus[~dfmus['Galaxy'].isin([3,4,11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,6,11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,4,6,11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([6,11,12])]

#ANALYSE
multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True)
#multigal.print_table()
multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],blind=False)
#multigal.plot_parameters(['mu','AV','theta'])
'''
multigal.plot_parameters(['mu','theta','AV'],g_or_z = 'g', subtract_g_mean=None)
multigal.plot_parameters('mu',g_or_z = 'g')
multigal.plot_parameters(['mu','theta','AV'],g_or_z = 'z')
multigal.plot_parameters('mu',g_or_z = 'z')
#'''
#multigal.n_warmup   = 2000 ; multigal.n_sampling = 8000
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=False,alt_prior=True,overwrite=True)
#multigal.plot_posterior_samples()
#'''
