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

'''#Inspect HRs/Flow Corrections
print (dfmus[dfmus['zhelio_errs']>=0.001][['SN','Galaxy','zhelio_hats','zhelio_errs','zHD_hats','zHD_errs']])
#print (dfmus[['SN','zhelio_hats','zhelio_errs','zHD_hats','zHD_errs']])
from astropy.cosmology import FlatLambdaCDM
cosmo  = FlatLambdaCDM(H0=73.24,Om0=0.28)
#dfmus = dfmus[dfmus['zhelio_errs']<0.001]
dfmus['muext_hats'] = dfmus[['zhelio_hats','zcmb_hats']].apply(lambda z: cosmo.distmod(z[1]).value + 5*np.log10((1+z[0])/(1+z[1])),axis=1)
dfmus['HRs'] = dfmus[['muext_hats','mus']].apply(lambda x: x[1]-x[0],axis=1)
#dfmus['HRs'] = dfmus[['mu_cosmicflows','mus']].apply(lambda x: x[1]-x[0],axis=1)
print (dfmus['HRs'].median(), dfmus['HRs'].std())

dfmus['muext_hats'] = dfmus[['zhelio_hats','zHD_hats']].apply(lambda z: cosmo.distmod(z[1]).value + 5*np.log10((1+z[0])/(1+z[1])),axis=1)
dfmus['HRs'] = dfmus[['muext_hats','mus']].apply(lambda x: x[1]-x[0],axis=1)
print (dfmus['HRs'].median(), dfmus['HRs'].std())
err=1/0
'''

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
multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],alpha_zhel=False,blind=True,save=True,show=False)

'''#Overlay analytic sigmaRel posteriors for different treatments of alpha_zhel
fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
pl.title(r'$\sigma_{\rm{Rel}}$-Posterior from Individual Distance Estimates',fontsize=multigal.FS+1, weight='bold')
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],blind=True,alpha_zhel=False,fig_ax=[fig,ax,0,r'Ignore $\mu(z_{\rm{Helio}})$ Dep.'],save=False,show=False)
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],blind=True,alpha_zhel=True,fig_ax=[fig,ax,1,r'Incl. $\mu(z_{\rm{Helio}})$ Dep.',kmax],save=True,show=False)
#'''


'''#Overlay analytic sigmaRel posteriors for different samples
fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
pl.title(r'$\sigma_{\rm{Rel}}$-Posterior from Individual Distance Estimates',fontsize=multigal.FS+1, weight='bold')
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],blind=True,fig_ax=[fig,ax,0,'Full Sample [12 Galaxies]'],save=False,show=False)
multigal.dfmus = multigal.dfmus[~multigal.dfmus['Galaxy'].isin([3,4,11,12])]
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],blind=True,fig_ax=[fig,ax,1,'Cosmo. Sample [8 Galaxies]',kmax],save=True,show=False)
#'''


#multigal.print_table()
#multigal.plot_parameters(['mu','AV','theta'])
'''#Plotting Parameters
multigal.plot_parameters(['mu','theta','AV'],g_or_z = 'g', subtract_g_mean=None)
multigal.plot_parameters('mu',g_or_z = 'g')
multigal.plot_parameters(['mu','theta','AV'],g_or_z = 'z')
#'''
'''#Plotting HRs
multigal.plot_parameters('mu',g_or_z = 'z',zword='zHD_hats')
multigal.plot_parameters('mu',g_or_z = 'z',zword='zcmb_hats')
multigal.plot_delta_HR()
err=1/0
#'''
#multigal.n_warmup   = 2000 ; multigal.n_sampling = 8000
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=False,alt_prior=True,overwrite=True,blind=True,zcosmo='zcmb',alpha_zhel=False)
multigal.plot_posterior_samples(blind=True)
#'''
