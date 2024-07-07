import sys

rootpath = './'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *

beta_prior_lines = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim \textrm{Arcsine}(0,1)$"]
alt_prior_lines  = [r"$\sigma_{\rm{Rel}} \sim U(0,1)$", r"$\sigma_{\rm{Common}} \sim U(0,1)$"]
priorA_lines     = [r"$\sigma_{0} \sim U(0,1)$", r"$\sigma_{\rm{Rel}} \sim U(0,\sigma_0)$"]
priorB_lines     = [r"$\sigma_{0} \sim U(0,1)$", r"$\sigma_{\rm{Common}} \sim U(0,\sigma_0)$"]
priorC_lines     = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim U(0,1)$"]

from simulate_distances import *

Ng = 100 ; Sg = 2 ; RS=10
#for rs in np.arange(RS):
tick_cross = dict(zip([True,False],['\\xmark','\\cmark']))
Summary_Strs = []
for rs in [9]:
    for sigR,RHO in zip([0,0.1/((2)**0.5),0.1],[1,0.5,0]):
        simulator = SiblingsDistanceSimulator(Ng=Ng,Sg=Sg,external_distances=True,sigmaRel=sigR,zcmberr=1e-5,random=42+rs)
        dfmus     = simulator.dfmus
        dfmus['zhelio_hats'] = dfmus['zcmb_hats']
        dfmus['zhelio_errs'] = dfmus['zcmb_errs']

        samplename = f'Ng{Ng}_Sg{Sg}_Rs{rs}_Truesigma0{simulator.sigma0}_TruesigmaRel{round(simulator.sigmaRel,3)}'
        multigal = multi_galaxy_siblings(dfmus,samplename=samplename,sigma0=1.0,eta_sigmaRel_input=None,sigmapec=250,use_external_distances=False)
        multigal.n_warmup = 2000 ; multigal.n_sampling = 10000

        #Multi-model overlay
        zcosmo = 'zcmb'
        #sigma0 = 0.1 ; lind = 1 ; Np = 4#For when sigma0 is fixed, don't plot sigR, sigC model
        sigma0 = 'free' ; lind = 0; Np = 4
        overwrite=True
        overwrite=False
        savekey   = f'multigalsims_{samplename}_Modelsigma0{sigma0}'

        counter = 0 ; print (savekey);
        recovery_str= tick_cross[RHO in []]
        multigal.sigmaRel_sampler(alt_prior=False,sigma0=sigma0,sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite)
        summary,postplot = multigal.plot_posterior_samples_1D(fig_ax=[None,    counter,Np,[0,-2+lind],''],returner=True); Summary_Strs.append(beta_prior_lines[-1]+' & \multirow{'+str(Np)+'}{*}{'+r'$\rho=%s$'%RHO+'} & '+' & '.join(list(summary.values())) + f'& {recovery_str} \\\\ ')
        counter += 1; print (savekey)

        #multigal.sigmaRel_sampler(alt_prior=True,sigma0=sigma0,sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite)
        #postplot = multigal.plot_posterior_samples_1D(fig_ax=[postplot,counter,Np,0,'',lind])
        #counter += 1; print (savekey)

        recovery_str=tick_cross[RHO in [0]]
        multigal.sigmaRel_sampler(alt_prior='A',sigma0=sigma0,sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite)
        summary,postplot = multigal.plot_posterior_samples_1D(fig_ax=[postplot,counter,Np,[0,0],'',lind],returner=True); Summary_Strs.append(priorA_lines[-1]+' & & '+' & '.join(list(summary.values())) +f'& {recovery_str} \\\\ ')
        counter += 1; print (savekey)

        recovery_str=tick_cross[RHO in [1]]
        multigal.sigmaRel_sampler(alt_prior='B',sigma0=sigma0,sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite)
        summary,postplot = multigal.plot_posterior_samples_1D(fig_ax=[postplot,counter,Np,[0,0],'',lind],returner=True); Summary_Strs.append(priorB_lines[-1]+' & & '+' & '.join(list(summary.values())) +f'& {recovery_str} \\\\ ')
        counter += 1; print (savekey)

        recovery_str=tick_cross[RHO in [0,1]]
        multigal.sigmaRel_sampler(alt_prior='C',sigma0=sigma0,sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite)
        #'''############################################################################################################################################
        #LINES DESCRIBING SIMULATIONS
        #Add common lines
        dy = (0.15-0.02*(len(postplot.parnames)<4))
        FS = 18-4 - 2*(len(postplot.parnames)<4)
        y0 = 0.05+dy*(len(postplot.parnames)-1)
        x0 = 0.95+1.1*(len(postplot.ax)==1)

        sig0true = simulator.sigma0 ; rho0true = simulator.rho ; sigRtrue = simulator.sigmaRel ; sigCtrue = simulator.sigmaCommon
        truth_dict = dict(zip(['sigma0','rho','sigmaRel','sigmaCommon'],[sig0true,rho0true,sigRtrue,sigCtrue]))
        truth_dict = {postplot.pardict[par]:truth_dict[par] for par in multigal.parnames}

        if sigR in [0,0.1/((2)**0.5),0.1]:
            pl.annotate(f"Constraining\n{dict(zip([0,0.1/((2)**0.5),0.1],['High', 'Intermediate','Low']))[sigR]}\nCorrelations", xy=(x0,y0+dy*2+0.45),xycoords='axes fraction',fontsize=FS+2,bbox=dict(facecolor='white',alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'),color='black',ha='right',weight='bold')

        pl.annotate('Simulation \nHyperparameters', xy=(x0,y0+dy*2),xycoords='axes fraction',fontsize=FS,color='black',ha='right',weight='bold')
        pl.annotate(postplot.lines[0], xy=(x0,y0+dy),xycoords='axes fraction',fontsize=FS,color='black',ha='right')
        for cntr,parlabel in enumerate(truth_dict):
            true_value = truth_dict[parlabel]
            units = parlabel.split('$')[-1]
            line  = r'%s = %s'%(parlabel if units=='' else parlabel.split(units)[0],round(true_value,3)) + units
            pl.annotate(line, xy=(x0,y0-dy*cntr),xycoords='axes fraction',fontsize=FS,color='black',ha='right')

            ymax = copy.deepcopy(postplot.ax[0,cntr].get_ylim()[1])
            print (true_value, ymax)
            postplot.ax[0,cntr].plot([true_value,true_value],[0,ymax],linewidth=5,color='black',zorder=100,linestyle='--')
            postplot.ax[0,cntr].set_ylim([0,ymax])
        #'''############################################################################################################################################
        summary, postplot = multigal.plot_posterior_samples_1D(fig_ax=[postplot,counter,Np,[0,0],'',lind],savekey=savekey,returner=True); Summary_Strs.append(priorC_lines[-1]+' & & '+' & '.join(list(summary.values())) + f'& {recovery_str} \\\\ ')
        Summary_Strs.append('\\midrule')
for x in Summary_Strs:
    print (x)
