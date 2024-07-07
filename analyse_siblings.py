import sys

rootpath = './'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *


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

#with open('products/multigal/FITNg100_Sg2_Rs9_Truesigma00.1_TruesigmaRel0free_sigma0_with_muextzcmb_fixed_sigmapec250_sigmaRelfree.pkl','rb') as f:
#    x = pickle.load(f)
#print (x['summary'].loc['sigmaRel'])
#err=1/0

#APPLY FILTERING
##COSMO SAMPLE
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,4,5,8])]


#ANALYSE
multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True)
#multigal.trim_sample()
'''#Print stats
print (multigal.dfmus[multigal.dfmus.zhelio_errs>0.001]['mu_errs'].mean(),multigal.dfmus[multigal.dfmus.zhelio_errs>0.001]['mu_errs'].std())
print (multigal.dfmus['mu_errs'].mean(),multigal.dfmus['mu_errs'].std())
dmus = [] ; sigs = []
for g in multigal.dfmus.Galaxy.unique():
    mus,muerrs = multigal.dfmus[multigal.dfmus.Galaxy==g][['mus','mu_errs']].T.values
    dmus.append(np.abs(mus[1]-mus[0]))
    sigs.append(dmus[-1]/np.sqrt(np.sum(np.square(muerrs))))
print (dmus)
print (sigs)
print (np.average(dmus),np.std(dmus),np.amin(dmus),np.amax(dmus))
print (np.average(sigs),np.std(sigs),np.amin(sigs),np.amax(sigs))
#err=1/0
multigal.print_table(PARS=['zhelio_hats','zhelio_errs','zcmb_hats','zHD_hats','mu','cosmo_sample'])
#multigal.print_table()
#multigal.plot_parameters(['mu','theta','AV'])#('mu')
multigal.plot_parameters('mu',g_or_z = 'z',zword='zHD_hats',annotate_mode=None,include_std=True)
multigal.plot_parameters('mu',g_or_z = 'z',zword='zcmb_hats',annotate_mode=None, include_std=True)
err=1/0
'''

multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],alpha_zhel=False,blind=False,save=True,show=False)
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,alt_prior=True,overwrite=False,blind=True,zcosmo='zcmb',alpha_zhel=True,zmarg=True)
#multigal.plot_posterior_samples(blind=True,show=False, save=True,pars=['sigmaRel','sigmaCommon'])


'''#Overlay analytic sigmaRel posteriors for different samples
fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
pl.title(r'$\sigma_{\rm{Rel}}$-Posterior from Individual Distance Estimates',fontsize=multigal.FS+1, weight='bold')
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(fig_ax=[fig,ax,0,'Full Sample [12 Galaxies]'],save=False,show=False)
multigal.trim_sample('cosmo')
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(fig_ax=[fig,ax,1,'Cosmo. Sub-sample [8 Galaxies]',kmax],save=True,show=False)
multigal.restore_sample()
#'''

'''#Posterior sampling to obtain cosmo-indep sigmaRel posterior
multigal.sigmaRel_sampler(sigma0=1,eta_sigmaRel_input=None,use_external_distances=False,overwrite=True,blind=False)
multigal.plot_posterior_samples()#pars=['sigmaRel','sigmaCommon'])
err=1/0
#'''

'''#Overlay analytic sigmaRel posteriors for different treatments of alpha_zhel
fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
pl.title(r'$\sigma_{\rm{Rel}}$-Posterior from Individual Distance Estimates',fontsize=multigal.FS+1, weight='bold')
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],alpha_zhel=False,fig_ax=[fig,ax,0,r'Ignore $\mu(z_{\rm{Helio}})$ Dep.'],save=False,show=False)
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],alpha_zhel=True, fig_ax=[fig,ax,1,r'Incl. $\mu(z_{\rm{Helio}})$ Dep.',kmax],save=True,show=False)
#'''

'''#Big corner plots
multigal.trim_sample()
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=False,zcosmo='zcmb')
multigal.plot_posterior_samples(pars=['rho','sigma0','sigmaRel','sigmaCommon','rel_rat','com_rat'],colour='C1',
    #filt_sigma0=[0.45,0.55],
    #filt_sigma0=[0.16,0.84]
    #filt_sigmaCommon=[0.16,0.84]
    )
err=1/0
#'''

'''#Quick cosmo-dep fits
#x4 fiducial cosmo-dep analyses
overwrite=False
#overwrite=True
colour='C0'
multigal.trim_sample() ; colour='C1'
Summary_Strs = []
for zcosmo in ['zcmb','zHD']:
    for sigmapec in [250,'free']:
        multigal.sigmaRel_sampler(sigma0='free',sigmapec=sigmapec,use_external_distances=True,overwrite=overwrite,zcosmo=zcosmo)
        summary = multigal.plot_posterior_samples(colour=colour,returner=True)
        Summary_Strs.append(' & '.join(list(summary.values())) + ' \\\\ ')
#Alternative rho hyperprior
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=overwrite,zcosmo='zcmb',alt_prior='C')
summary = multigal.plot_posterior_samples(returner=True)
Summary_Strs.append(' & '.join(list(summary.values())) + ' \\\\ ')

for x in Summary_Strs:
    print (x)
err=1/0
#'''

#'''#cosmodep overlays full and cosmo-subsample
overwrite=False
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=overwrite,zcosmo='zcmb')
postplot = multigal.plot_posterior_samples_1D(FS=20,pars=['rho','rel_rat2','com_rat2'],fig_ax=[None,0,2,[1,None],['Full Sample','[12 Galaxies]']],show=False, save=False,lines=True)

multigal.trim_sample()
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=overwrite,zcosmo='zcmb')
postplot = multigal.plot_posterior_samples_1D(FS=20,pars=['rho','rel_rat2','com_rat2'],fig_ax=[postplot,1,2,[1,None],['Cosmology Sub-sample','[8 Galaxies]']],show=False, save=True,lines=True)

#multigal.restore_sample()
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=overwrite,zcosmo='zcmb')
#postplot = multigal.plot_posterior_samples_1D(FS=20,pars=['sigma0','sigmaRel','sigmaCommon'],fig_ax=[None,0,2,[1,None],['Full Sample','[12 Galaxies]']],show=False, save=False,lines=False)
#multigal.trim_sample()
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=overwrite,zcosmo='zcmb')
#postplot = multigal.plot_posterior_samples_1D(FS=20,pars=['sigma0','sigmaRel','sigmaCommon'],fig_ax=[postplot,1,2,[1,None],['Cosmology Sub-sample','[8 Galaxies]']],show=False, save=True,lines=False)
#'''

'''#Long Cosmo-dep fits
#Appendices (z-marginalisation)
overwrite = False
#multigal.n_warmup = 2000 ; multigal.n_sampling = 10000 ; overwrite=True
Summary_Strs = []
for zcosmo in ['zcmb']:
    for alpha_zhel in [False,True]:
        multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=True,overwrite=overwrite,zcosmo=zcosmo,alpha_zhel=alpha_zhel)
        summary = multigal.plot_posterior_samples(returner=True)
        Summary_Strs.append(' & '.join(list(summary.values())) + ' \\\\ ')

for x in Summary_Strs:
    print (x)
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


'''#Test sens to hyperpriors
zcosmo = 'zcmb'
overwrite=False

arcsine_lines = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim \rm{Arcsine}(0,1)$"]
#alt_prior_lines  = [r"$\sigma_{\rm{Rel}} \sim U(0,1)$", r"$\sigma_{\rm{Common}} \sim U(0,1)$"]
#priorA_lines     = [r"$\sigma_{0} \sim U(0,1)$", r"$\sigma_{\rm{Rel}} \sim U(0,\sigma_0)$"]
#priorB_lines     = [r"$\sigma_{0} \sim U(0,1)$", r"$\sigma_{\rm{Common}} \sim U(0,\sigma_0)$"]
priorC_lines     = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim U(0,1)$"]

multigal.sigmaRel_sampler(alt_prior=False,sigma0='free',sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite)
postplot = multigal.plot_posterior_samples(fig_ax=[None,0,2,-2,arcsine_lines],show=False, save=False, quick=False)
postplot = multigal.plot_posterior_samples(fig_ax=[postplot,0,2,-2,['']],show=False, save=False, quick=True)
multigal.sigmaRel_sampler(alt_prior='C',sigma0='free',sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite)
postplot = multigal.plot_posterior_samples(fig_ax=[postplot,1,2,0,priorC_lines],show=False,save=False, quick=True)
#Add common lines
dy  = (0.15-0.02*(len(postplot.parnames)<4)) ; yy0 = postplot.y0-0.35+0.06*(len(postplot.parnames)<4) ; FS  = 18-4 - 2*(len(postplot.parnames)<4)
counter = -1
for ticker,line in enumerate(postplot.lines):
    if ticker in [0,3,4]:
        counter += 1
        pl.annotate(line, xy=(1+1.1*(len(postplot.ax)==1),yy0-dy*(counter-1)),xycoords='axes fraction',fontsize=FS,color='black',ha='right')
postplot = multigal.plot_posterior_samples(fig_ax=[postplot,1,2,0,['']],show=False,save=True, quick=False,blind=True)
#'''



'''#Multi-model overlay (mu, z, z w/ mu-zhelio) 1D OVERLAY
overwrite=False
#multigal.n_warmup   = 3000 ; multigal.n_sampling = 14000 ; overwrite=True
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=False,alt_prior=True,overwrite=overwrite,blind=True,zcosmo='zcmb',alpha_zhel=False)

postplot = multigal.plot_posterior_samples_1D(blind=True, fig_ax=[None,0,3,-1,[r'Modelled $\hat{z}_{\rm{CMB}}$ Distances']],show=False, save=False, quick=True, lines=True)
postplot = multigal.plot_posterior_samples_1D(blind=True, fig_ax=[postplot,0,3,-1,['']],show=False, save=False, quick=False,lines=True)

multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=True,alt_prior=True,overwrite=overwrite,blind=True,zcosmo='zcmb',alpha_zhel=False)
postplot = multigal.plot_posterior_samples_1D(blind=True, fig_ax=[postplot,1,3,None,[r'Modelled $z_{\rm{CMB}}$ Parameters']],show=False, save=False, quick=False,lines=True)

multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=True,alt_prior=True,overwrite=overwrite,blind=True,zcosmo='zcmb',alpha_zhel=True)
postplot = multigal.plot_posterior_samples_1D(blind=True, fig_ax=[postplot,2,3,-2,[r'Modelled $z_{\rm{CMB}}$ Parameters +',r'$\epsilon_{\mu} = \hat{\alpha} \epsilon_{z_{\rm{Helio}}}$ for 3 Galaxies']],show=False, save=True, quick=False,lines=True)
#'''

'''#Multi-model overlay (mu, z, z w/ mu-zhelio) 2D CORNER PLOT
overwrite=False
#multigal.n_warmup   = 2000 ; multigal.n_sampling = 10000 ; overwrite=True

multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=False,alt_prior='C',overwrite=overwrite,blind=True,zcosmo='zcmb',alpha_zhel=False)

postplot = multigal.plot_posterior_samples(blind=True, fig_ax=[None,0,3,-1,[r'Modelled $\hat{z}_{\rm{CMB}}$ Distances']],show=False, save=False, quick=True)
postplot = multigal.plot_posterior_samples(blind=True, fig_ax=[postplot,0,3,-1,['']],show=False, save=False, quick=False)

multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=True,alt_prior='C',overwrite=overwrite,blind=True,zcosmo='zcmb',alpha_zhel=False)
postplot = multigal.plot_posterior_samples(blind=True, fig_ax=[postplot,1,3,None,[r'Modelled $z_{\rm{CMB}}$ Parameters']],show=False, save=False, quick=False)

multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=True,alt_prior='C',overwrite=overwrite,blind=True,zcosmo='zcmb',alpha_zhel=True)
postplot = multigal.plot_posterior_samples(blind=True, fig_ax=[postplot,2,3,-2,[r'Modelled $z_{\rm{CMB}}$ Parameters +',r'$\epsilon_{\mu} = \hat{\alpha} \epsilon_{z_{\rm{Helio}}}$ for 3 Galaxies']],show=False, save=True, quick=False)
#'''
