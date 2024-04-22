import sys

rootpath = './'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *

'''#Simulated Data
from simulate_distances import *
simulator = SiblingsDistanceSimulator(Ng=100,Sg=3,external_distances=True,sigmaRel=0.09999,zcmberr=1e-5)
dfmus     = simulator.dfmus
dfmus['zhelio_hats'] = dfmus['zcmb_hats']
dfmus['zhelio_errs'] = dfmus['zcmb_errs']
print (dfmus)
#multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec='free',use_external_distances=True)
#multigal = multi_galaxy_siblings(dfmus,sigma0='free',eta_sigmaRel_input=None,sigmapec=250,use_external_distances=True)
multigal = multi_galaxy_siblings(dfmus,sigma0=1.0,eta_sigmaRel_input=None,sigmapec=250,use_external_distances=False)
#multigal.compute_analytic_multi_gal_sigmaRel_posterior()
#multigal.sigmaRel_sampler()#sigma0=None, sigmapec=None, eta_sigmaRel_input=None, use_external_distances=None, overwrite=True)
#multigal.plot_posterior_samples()
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,alt_prior=False)
#multigal.plot_posterior_samples()
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,alt_prior=True)
#multigal.plot_posterior_samples()
multigal.sigmaRel_sampler(sigma0='free',sigmapec='free',use_external_distances=True,zmarg=False,overwrite=True,alt_prior=True)
multigal.plot_posterior_samples()
multigal.sigmaRel_sampler(sigma0='free',sigmapec='free',use_external_distances=True,zmarg=True, overwrite=True,alt_prior=True)
multigal.plot_posterior_samples()

err=1/0
#'''


from load_data import *
dfmus = load_dfmus('ZTFtest5')

#print (dfmus[dfmus['zhelio_errs']==0.01][['SN','zhelio_hats','zhelio_errs','zHD_hats','zHD_errs']])
print (dfmus[['SN','zhelio_hats','zhelio_errs','zHD_hats','zHD_errs']])


#APPLY FILTERING
##COSMO SAMPLE
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,4,11,12])]

#dfmus = dfmus[~dfmus['Galaxy'].isin([1,3,6,11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,6,11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,4,6,11,12])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([6,11,12])]

#ANALYSE
multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True)

multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],blind=True,save=True,show=False)


'''#Overlay analytic sigmaRel posteriors for different samples
fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
pl.title(r'$\sigma_{\rm{Rel}}$-Posterior from Individual Distance Estimates',fontsize=multigal.FS+1, weight='bold')
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],blind=True,fig_ax=[fig,ax,0,'Full Sample [12 Galaxies]'],save=False,show=False)
multigal.dfmus = multigal.dfmus[~multigal.dfmus['Galaxy'].isin([3,4,11,12])]
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],blind=True,fig_ax=[fig,ax,1,'Cosmo. Sample [8 Galaxies]',kmax],save=True,show=False)
#'''



#multigal.print_table()
#multigal.plot_parameters(['mu','AV','theta'])
'''
multigal.plot_parameters(['mu','theta','AV'],g_or_z = 'g', subtract_g_mean=None)
multigal.plot_parameters('mu',g_or_z = 'g')
multigal.plot_parameters(['mu','theta','AV'],g_or_z = 'z')
#'''
#multigal.plot_parameters('mu',g_or_z = 'z',zword='zHD_hats')
multigal.plot_parameters('mu',g_or_z = 'z',zword='zcmb_hats')
err=1/0
#multigal.n_warmup   = 2000 ; multigal.n_sampling = 8000
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=False,alt_prior=True,overwrite=True,blind=True)
multigal.plot_posterior_samples(blind=True)
#'''
