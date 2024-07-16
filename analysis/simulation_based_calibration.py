import sys

rootpath = '../'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *
from simulate_distances import *

arcsine_prior_lines = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim \textrm{Arcsine}(0,1)$"]
priorC_lines        = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim U(0,1)$"]

#PERFORM SBC
Ng = 100 ; Sg = 2 ; RS=100
zcosmo = 'zcmb'
sigma0 = 'free'
Rhat_threshold = 1.02 ; Ntrials=10
tick_cross = dict(zip([True,False],['\\xmark','\\cmark']))
Summary_Strs = []

overwrite = True
overwrite = False

ALT_PRIOR = False
ALT_PRIOR = 'C'

productpath = None; modelkey  = None; sim_sigma0 = None
for rs in np.arange(RS):
    for sigR,RHO in zip([0,0.1/((2)**0.5),0.1],[1,0.5,0]):
        try:
            if modelkey is not None and overwrite is False:
                samplename = f'Ng{Ng}_Sg{Sg}_Rs{rs}_Truesigma0{sim_sigma0}_TruesigmaRel{round(sigR,3)}'
                filename   = productpath+f"FIT{samplename}{modelkey}.pkl"
                if os.path.exists(filename):
                    pass
                else:
                    raise Exception()
            else:
                raise Exception()
        except:
            print ((('#~#~#'*30)+'\n')*3)
            print (f'Performing: random_state={rs}/{RS}; sigR={sigR} equiv rho={RHO}')
            print ('\n'+(('#-#-#'*30)+'\n')*3)
            simulator  = SiblingsDistanceSimulator(Ng=Ng,Sg=Sg,external_distances=True,sigmaRel=sigR,zcmberr=1e-5,random=42+rs)
            dfmus     = simulator.dfmus
            dfmus['zhelio_hats'] = dfmus['zcmb_hats']
            dfmus['zhelio_errs'] = dfmus['zcmb_errs']

            samplename = f'Ng{Ng}_Sg{Sg}_Rs{rs}_Truesigma0{simulator.sigma0}_TruesigmaRel{round(simulator.sigmaRel,3)}'
            multigal = multi_galaxy_siblings(dfmus,samplename=samplename,sigma0=1.0,eta_sigmaRel_input=None,sigmapec=250,use_external_distances=False,rootpath=rootpath)
            multigal.n_warmup = 250 ; multigal.n_sampling = 250 ; multigal.n_thin = 250

            #Multi-model overlay
            multigal.sigmaRel_sampler(alt_prior=ALT_PRIOR,sigma0=sigma0,sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite,Rhat_threshold=Rhat_threshold, Ntrials=Ntrials)

            #savekey   = f'multigalsims_{samplename}_Modelsigma0{sigma0}'#For plots
            if modelkey is None:
                productpath = copy.deepcopy(multigal.productpath)
                sim_sigma0  = copy.deepcopy(simulator.sigma0)
                modelkey    = copy.deepcopy(multigal.modelkey)
                multigal.get_parlabels(['rho','sigma0','sigmaRel','sigmaCommon'])
                dfpars    = dict(zip(multigal.parnames,multigal.dfparnames))
                parlabels = dict(zip(multigal.parnames, [x.replace('$','').replace(' (mag)','') for x in multigal.parlabels]))
                bounds    = dict(zip(multigal.parnames, multigal.bounds))


import argparse,yaml
from sbc_plot_functions import *
parser = argparse.ArgumentParser(description="SBC Input Hyperparameters")
parser.add_argument("--loop_par",	      default='sigmaRel',         help='Set parameter to loop over')
parser.add_argument("--sigmaRel",	      default=0.1, 	              help='Float for non-loop sigmaRel')
parser.add_argument("--loop_sigmaRel",    default='1.5,2.5,3.5',      help='Parameter values to loop over')
parser.add_argument("--save",	default=True,		      help='Save plot')
parser.add_argument("--show",	default=False,		      help='Show plot')
parser.add_argument("--quantilemode",	default=True,	  help='If True, annotate with 16,84, False, use sample std.')
args = parser.parse_args().__dict__
loop_par      = args['loop_par']
loop_par_dict = {loop_par:[float(s) for s in args[f"loop_{args['loop_par']}"].split(',')]}

#PLOT SBC
Nsim_keep = 100
GLOB_FITS = {}
for sigR,RHO in zip([0,0.1/((2)**0.5),0.1],[1,0.5,0]):
    FITS = {}
    for ISIM in np.arange(RS):
        samplename = f'Ng{Ng}_Sg{Sg}_Rs{ISIM}_Truesigma0{sim_sigma0}_TruesigmaRel{round(sigR,3)}'
        filename   = productpath+f"FIT{samplename}{modelkey}.pkl"
        with open(filename,'rb') as f:
            FIT = pickle.load(f)
        FITS[ISIM] = FIT
        #print (FIT['chains'].shape[0])
    FITS = trim_to_KEEPERS({'dummy':FITS},get_KEEPERS({'dummy':FITS},Nsim_keep,Rhat_threshold,loop_par,dfpars[loop_par]))['dummy']
    GLOB_FITS[RHO] = FITS


fig,axs = pl.subplots(len(GLOB_FITS),1,figsize=(8,6*len(GLOB_FITS)),sharex=False)
for iax,true_par in enumerate(GLOB_FITS):
	FITS    = GLOB_FITS[true_par]
	plotter = SBC_FITS_PLOTTER(iax,fig.axes,[true_par,loop_par,dfpars[loop_par],parlabels[loop_par]],FITS,bounds,rootpath,quantilemode=args['quantilemode'])
	plotter.plot_sbc_panel()
#title_tuples = [l for key,value in non_loop_pars.items() for l in [parlabels[key],value]]
#fig.axes[0].set_title('Fits to SED-Integrated Simulated Data;\nTrue Simulation Parameters: '+r'$%s = %s ; %s = %s\,$mag'%(title_tuples[0],title_tuples[1],title_tuples[2],title_tuples[3]),fontsize=plotter.FS)
fig.axes[-1].set_xlabel(r'%s'%parlabels[loop_par],fontsize=plotter.FS)
fig.axes[0].set_ylabel('Posterior Densities',fontsize=plotter.FS,color='white')#For spacing
fig.text(0.01, 0.5, 'Posterior Densities', ha='center', va='center', rotation='vertical',fontsize=plotter.FS)
pl.tight_layout()
#if args['save']:
#	pl.savefig(f"{sbc.plotpath}SBC_looppar{loop_par}_nonlooppars{non_loop_pars}_quantilemode{args['quantilemode']}.pdf",bbox_inches='tight')
#if args['show']:
pl.show()


#Sam Return on July 15th 2024; now use sbc_plot_functions.py to plot up SBC results
