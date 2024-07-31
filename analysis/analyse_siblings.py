import sys

rootpath = '../'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *


from load_data import *
dfmus = load_dfmus('ZTFtest5')

#Loading data with limited columns included
multigal   = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True,rootpath=rootpath,verbose=False)
keep_cols  = ['SN','Galaxy','mus','mu_errs','zhelio_hats','zhelio_errs','zcmb_hats','zHD_hats','zcmb_errs','zHD_errs','cosmo_sample','alpha_mu_z']
dfmus      = {key:multigal.dfmus[key].values for key in keep_cols}

#ANALYSE
multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True,rootpath=rootpath)

'''#Print sample statistics, distance posteriors and Hubble diagrams
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
multigal.print_table(PARS=['zhelio_hats','zhelio_errs','zcmb_hats','zHD_hats','mu','cosmo_sample'])
#multigal.plot_parameters()
#multigal.plot_parameters('mu',g_or_z = 'z',zword='zHD_hats',annotate_mode=None,include_std=True)
#multigal.plot_parameters('mu',g_or_z = 'z',zword='zcmb_hats',annotate_mode=None, include_std=True)
#multigal.plot_delta_HR()
err=1/0
#'''

'''#Cosmo-indep sigR posterior
multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],save=True,show=False)
#'''

'''#Cosmo-indep sigR posterior: sensitivity to different samples
fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
pl.title(r'$\sigma_{\rm{Rel}}$-Posterior from Individual Distance Estimates',fontsize=multigal.FS+1, weight='bold')
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(fig_ax=[fig,ax,0,'Full Sample [12 Galaxies]'],save=False,show=False)
multigal.trim_sample('cosmo')
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(fig_ax=[fig,ax,1,'Cosmo. Sub-sample [8 Galaxies]',kmax],save=True,show=False)
multigal.restore_sample()
#'''

'''#Cosmo-indep sigR posterior: sensitivity to sigR hyperprior
fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
pl.title(r'$\sigma_{\rm{Rel}}$-Posterior from Individual Distance Estimates',fontsize=multigal.FS+1, weight='bold')
multigal.trim_sample('cosmo')
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(fig_ax=[fig,ax,0,'Cosmo. Sub-sample [Uniform]'],prior='uniform',save=False,show=False)
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(fig_ax=[fig,ax,1,'Cosmo. Sub-sample [Jeffreys]',kmax],prior='jeffreys',save=False,show=False)
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(fig_ax=[fig,ax,2,'Cosmo. Sub-sample [p2]',kmax],prior='p2',save=True,show=False)
multigal.restore_sample()
#'''

'''#Cosmo-indep sigR posterior: sensitivity to treatment of heliocentric redshift errors
fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
pl.title(r'$\sigma_{\rm{Rel}}$-Posterior from Individual Distance Estimates',fontsize=multigal.FS+1, weight='bold')
multigal.trim_sample()
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],alpha_zhel=False,fig_ax=[fig,ax,0,r'Ignore $\mu(z_{\rm{Helio}})$ Dep.'],save=False,show=False)
fig,ax,kmax = multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],alpha_zhel=True, fig_ax=[fig,ax,1,r'Incl. $\mu(z_{\rm{Helio}})$ Dep.',kmax],save=True,show=False)
multigal.restore_sample()
#'''

'''#Cosmo-indep sigR posterior: posterior sampling method (sigma0 fixed at 1, using Arcsine rho-hyperprior)
multigal.sigmaRel_sampler(sigma0=1,eta_sigmaRel_input=None,use_external_distances=False)
multigal.plot_posterior_samples()
#'''

'''#Cosmo-dep fits: sensitivity to zcosmo, sigmapec, cosmo sub-samp, and rho-hyperprior
overwrite=False
#overwrite=True
Summary_Strs = []
for cosmo_subsamp,colour in zip([False,True],['C0','C1']):
    if cosmo_subsamp:   multigal.trim_sample()
    else:               multigal.restore_sample()
    for zcosmo in ['zcmb','zHD']:
        for sigmapec in [250,'free']:
            multigal.sigmaRel_sampler(sigma0='free',sigmapec=sigmapec,use_external_distances=True,overwrite=overwrite,zcosmo=zcosmo)
            summary = multigal.plot_posterior_samples(colour=colour,returner=True)
            Summary_Strs.append(' & '.join(list(summary.values())) + ' \\\\ ')
    #Alternative uniform rho-hyperprior
    multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=overwrite,zcosmo='zcmb',alt_prior='C')
    summary = multigal.plot_posterior_samples(colour=colour,returner=True)
    Summary_Strs.append(' & '.join(list(summary.values())) + ' \\\\ ')

for x in Summary_Strs:
    print (x)
#'''

'''#Cosmo-dep fit plotting: overlay 1D marginal posteriors for full sample and cosmo-subsample
#rho, rel_rat, com_rat
overwrite=False
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=overwrite,zcosmo='zcmb')
postplot = multigal.plot_posterior_samples_1D(FS=20,pars=['rho','rel_rat','com_rat'],fig_ax=[None,0,2,[1,None],['Full Sample','[12 Galaxies]']],show=False, save=False,lines=True)
multigal.trim_sample()
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=overwrite,zcosmo='zcmb')
postplot = multigal.plot_posterior_samples_1D(FS=20,pars=['rho','rel_rat','com_rat'],fig_ax=[postplot,1,2,[1,None],['Cosmology Sub-sample','[8 Galaxies]']],show=False, save=True,lines=True)

#sig0, sigR, sigC
#multigal.restore_sample()
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=overwrite,zcosmo='zcmb')
#postplot = multigal.plot_posterior_samples_1D(FS=20,pars=['sigma0','sigmaRel','sigmaCommon'],fig_ax=[None,0,2,[1,None],['Full Sample','[12 Galaxies]']],show=False, save=False,lines=False)
#multigal.trim_sample()
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=overwrite,zcosmo='zcmb')
#postplot = multigal.plot_posterior_samples_1D(FS=20,pars=['sigma0','sigmaRel','sigmaCommon'],fig_ax=[postplot,1,2,[1,None],['Cosmology Sub-sample','[8 Galaxies]']],show=False, save=True,lines=False)
#'''

'''#Cosmo-dep fits: z-pipeline fits, and option to marginalise over heliocentric redshift for galaxies with sigmaz=0.01
overwrite = False
#multigal.n_warmup = 2000 ; multigal.n_sampling = 10000 ; overwrite=True
Summary_Strs = []
for zcosmo in ['zcmb']:
    for alpha_zhel in [False,True]:
        multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,
                                 zmarg=True,overwrite=overwrite,zcosmo=zcosmo,alpha_zhel=alpha_zhel)
        summary = multigal.plot_posterior_samples(returner=True)
        Summary_Strs.append(' & '.join(list(summary.values())) + ' \\\\ ')

for x in Summary_Strs:
    print (x)
#'''

'''#Cosmo-dep fit plotting: option to show more parameters and/or filter posterior samples
multigal.trim_sample()
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,overwrite=False,zcosmo='zcmb')
multigal.plot_posterior_samples(pars=['rho','sigma0','sigmaRel','sigmaCommon','rel_rat','com_rat'],colour='C1',
    #filt_sigma0=[0.45,0.55],
    #filt_sigma0=[0.16,0.84]
    #filt_sigmaCommon=[0.16,0.84]
    )
#'''

'''#Cosmo-dep fit plotting: 1D marginal posterior overlay for x3 choices of pipeline (mu, z, z w/ mu-zhelio)
overwrite=False
#multigal.n_warmup   = 3000 ; multigal.n_sampling = 14000 ; overwrite=True
multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=False,overwrite=overwrite,zcosmo='zcmb')
postplot = multigal.plot_posterior_samples_1D(fig_ax=[None,0,3,[None,-1],[r'Modelled $\hat{z}_{\rm{CMB}}$ Distances']],show=False, save=False, lines=True)

multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=True,overwrite=overwrite,zcosmo='zcmb',alpha_zhel=False)
postplot = multigal.plot_posterior_samples_1D(fig_ax=[postplot,1,3,[None,None],[r'Modelled $z_{\rm{CMB}}$ Parameters']],show=False, save=False,lines=True)

multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True,zmarg=True,overwrite=overwrite,zcosmo='zcmb',alpha_zhel=True)
postplot = multigal.plot_posterior_samples_1D(fig_ax=[postplot,2,3,[None,-2],[r'Modelled $z_{\rm{CMB}}$ Parameters +',r'$\epsilon_{\mu} = \hat{\alpha} \epsilon_{z_{\rm{Helio}}}$ for 3 Galaxies']],show=False, save=True,lines=True)
#'''

'''#Cosmo-dep fit plotting: 2D corner plot overlaying posteriors for different choices of rho-hyperprior
zcosmo = 'zcmb'
overwrite=False

arcsine_lines = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim \rm{Arcsine}(0,1)$"]
priorC_lines  = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim U(0,1)$"]

multigal.sigmaRel_sampler(alt_prior=False,sigma0='free',sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite)
postplot = multigal.plot_posterior_samples(fig_ax=[None,0,2,[None,-2],arcsine_lines],show=False, save=False, quick=False)
postplot = multigal.plot_posterior_samples(fig_ax=[postplot,0,2,[None,-2],['']],show=False, save=False, quick=True)

multigal.sigmaRel_sampler(alt_prior='C',sigma0='free',sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite)
postplot = multigal.plot_posterior_samples(fig_ax=[postplot,1,2,[None,0],priorC_lines],show=False,save=False, quick=True)
#Add common lines
dy  = (0.15-0.02*(len(postplot.parnames)<4)) ; yy0 = postplot.y0-0.35+0.06*(len(postplot.parnames)<4) ; FS  = 18-4 - 2*(len(postplot.parnames)<4)
counter = -1
for ticker,line in enumerate(postplot.lines):
    if ticker in [0,3,4]:
        counter += 1
        pl.annotate(line, xy=(1+1.1*(len(postplot.ax)==1),yy0-dy*(counter-1)),xycoords='axes fraction',fontsize=FS,color='black',ha='right')
postplot = multigal.plot_posterior_samples(fig_ax=[postplot,1,2,[None,0],['']],show=False,save=True, quick=False)
#'''



'''#Alternative fits: free choice of sampling from some p-plane
multigal.trim_sample()
multigal.p = 1
multigal.usamp_input = 0
multigal.samplename += f'_p{multigal.p}_usamp{multigal.usamp_input}'
#multigal.n_warmup = 100000
#multigal.n_sampling = 10000
multigal.sigmaRel_sampler(alt_prior='p',sigma0='free',sigmapec=250,use_external_distances=True,zcosmo='zcmb')
multigal.plot_posterior_samples()#show=False, save=True,pars=['sigmaRel','sigmaCommon'])
multigal.samplename = multigal.samplename.replace(f'_p{multigal.p}_usamp{multigal.usamp_input}','')
multigal.restore_sample()
#'''

'''#Old Code; Inspect HRs/Flow Corrections
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
#'''
