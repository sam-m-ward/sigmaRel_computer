"""
python simulation_based_calibration.py --looppar rho --altprior False
python simulation_based_calibration.py --looppar sigmaRel --altprior False
python simulation_based_calibration.py --looppar sigmaCommon --altprior False
python simulation_based_calibration.py --looppar rel_rat --altprior False
python simulation_based_calibration.py --looppar com_rat --altprior False

python simulation_based_calibration.py --looppar rho --altprior C
python simulation_based_calibration.py --looppar sigmaRel --altprior C
python simulation_based_calibration.py --looppar sigmaCommon --altprior C
python simulation_based_calibration.py --looppar rel_rat --altprior C
python simulation_based_calibration.py --looppar com_rat --altprior C
"""
import argparse,sys,yaml
rootpath = '../'
sys.path.append(rootpath+'model_files/')
from sbc_plot_functions import *
from sigmaRel import *
from simulate_distances import *

#SBC settings
Nsim_keep = 100
Ng = 100 ; Sg = 2 ; RS=100 ; sigma_fit_s = 0.05 ; ssigma0 = 0.1
#sigma_fit_s = 0.1 ; ssigma0=0.15
zcosmo = 'zcmb'
sigma0 = 'free'
Rhat_threshold = 1.02 ; Ntrials=10
overwrite = True
overwrite = False

#Initialise args
parser = argparse.ArgumentParser(description="SBC Input Hyperparameters")
parser.add_argument("--looppar",	      default='sigmaRel',         help='Set parameter to loop over')
parser.add_argument("--altprior",	      default=False,         	  help='Set alt_prior')
parser.add_argument("--save",	default=True,		      help='Save plot')
parser.add_argument("--show",	default=False,		      help='Show plot')
parser.add_argument("--quantilemode",	default=True,	  help='If True, annotate with 16,84, False, use sample std.')

#Read args
args = parser.parse_args().__dict__
loop_par  = args['looppar']
try:	ALT_PRIOR = {'False':False,'True':True}[args['altprior']]
except:	ALT_PRIOR = args['altprior']
if ALT_PRIOR == False:	hyperprior_lines = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim \rm{Arcsine}(0,1)$"]
if ALT_PRIOR == 'C': 	hyperprior_lines = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim U(0,1)$"]

#Perform SBC
productpath = None; plotpath = None; modelkey  = None; sim_sigma0 = None
for rs in np.arange(RS):
	for sigR,RHO in zip([0,ssigma0/((2)**0.5),ssigma0],[1,0.5,0]):
		try:
			if modelkey is not None and overwrite is False:
				samplename = f'Ng{Ng}_Sg{Sg}_Rs{rs}_Truesigma0{sim_sigma0}_TruesigmaRel{round(sigR,3)}'+(sim_sigmafit!=0.05)*f'_sigmafit{sim_sigmafit}'
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
			simulator  = SiblingsDistanceSimulator(Ng=Ng,Sg=Sg,external_distances=True,sigmaRel=sigR,zcmberr=1e-5,random=42+rs,sigma_fit_s=sigma_fit_s,sigma0=ssigma0)
			dfmus     = simulator.dfmus
			dfmus['zhelio_hats'] = dfmus['zcmb_hats']
			dfmus['zhelio_errs'] = dfmus['zcmb_errs']

			samplename = f'Ng{Ng}_Sg{Sg}_Rs{rs}_Truesigma0{simulator.sigma0}_TruesigmaRel{round(simulator.sigmaRel,3)}'+(simulator.sigma_fit_s[0]!=0.05)*f'_sigmafit{simulator.sigma_fit_s[0]}'
			multigal = multi_galaxy_siblings(dfmus,samplename=samplename,sigma0=1.0,eta_sigmaRel_input=None,sigmapec=250,use_external_distances=False,rootpath=rootpath)
			multigal.n_warmup = 250 ; multigal.n_sampling = 250 ; multigal.n_thin = 250

			#Multi-model overlay
			multigal.sigmaRel_sampler(alt_prior=ALT_PRIOR,sigma0=sigma0,sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite,Rhat_threshold=Rhat_threshold, Ntrials=Ntrials)

			if modelkey is None:
				productpath = copy.deepcopy(multigal.productpath)
				plotpath    = copy.deepcopy(multigal.plotpath)+'sbc/'
				if not os.path.exists(plotpath):    os.mkdir(plotpath)
				sim_sigma0   = copy.deepcopy(simulator.sigma0)
				sim_sigmafit = copy.deepcopy(simulator.sigma_fit_s[0])
				modelkey     = copy.deepcopy(multigal.modelkey)
				multigal.get_parlabels(['rho','sigma0','sigmaRel','sigmaCommon','rel_rat','com_rat','rel_rat2','com_rat2'])
				dfpars         = dict(zip(multigal.parnames,multigal.dfparnames))
				parlabels      = dict(zip(multigal.parnames, [x.replace('$','').replace(' (mag)','') for x in multigal.parlabels]))
				parlabels_full = dict(zip(multigal.parnames, multigal.parlabels))
				bounds         = dict(zip(multigal.parnames, multigal.bounds))

vals = {'rho':[1,0.5,0],
		'sigma0':[sim_sigma0,sim_sigma0,sim_sigma0],
		'sigmaRel'   :[0,sim_sigma0/((2)**0.5),sim_sigma0],
		'sigmaCommon':[sim_sigma0,sim_sigma0/((2)**0.5),0],
		'rel_rat':[0,1/((2)**0.5),1],
		'com_rat':[1,1/((2)**0.5),0],
		'rel_rat2':[0,0.5,1],
		'com_rat2':[1,0.5,0]}
df_vals = pd.DataFrame(vals)

#Collate Results
GLOB_FITS = {}
for sigR,RHO in zip(df_vals.sigmaRel.values,df_vals.rho.values):
	FITS = {}
	for ISIM in np.arange(RS):
		samplename = f'Ng{Ng}_Sg{Sg}_Rs{ISIM}_Truesigma0{sim_sigma0}_TruesigmaRel{0 if sigR==0 else round(sigR,3)}'+(sim_sigmafit!=0.05)*f'_sigmafit{sim_sigmafit}'
		filename   = productpath+f"FIT{samplename}{modelkey}.pkl"
		with open(filename,'rb') as f:
			FIT = pickle.load(f)
		chains, summary = multigal.add_transformed_params(FIT['chains'], FIT['summary'])#Add transformed parameters
		FIT['chains'] = chains; FIT['summary'] = summary
		FITS[ISIM] = FIT
	FITS = trim_to_KEEPERS({'dummy':FITS},get_KEEPERS({'dummy':FITS},Nsim_keep,Rhat_threshold,loop_par,dfpars[loop_par]))['dummy']
	GLOB_FITS[RHO] = FITS

###Plot SBC
#Plot of recovery of args['looppar']
fig,axs = pl.subplots(len(GLOB_FITS),1,figsize=(8,6*len(GLOB_FITS)),sharex=False)
for iax,true_rho in enumerate(GLOB_FITS):
	FITS      = GLOB_FITS[true_rho]
	sap,Ncred = get_sap_Ncred(loop_par,true_rho,ALT_PRIOR)
	true_par  = round(df_vals[df_vals.rho==true_rho][loop_par].values[0],3)
	plotter   = SBC_FITS_PLOTTER(iax,fig.axes,[true_par,loop_par,dfpars[loop_par],parlabels[loop_par]],FITS,bounds,rootpath,quantilemode=args['quantilemode'])
	plotter.plot_sbc_panel(sap=sap, Ncred = Ncred, Lside = True if ('_rat' in loop_par and true_rho==0.5) else False,FAC = 25)
fig.axes[0].set_title('Fits to %s$\\times$%s simulations of %s sibling-pair galaxies;\nTrue $\sigma_0 = %s\,$mag;\nHyperpriors: %s'%(df_vals.shape[0],Nsim_keep,Ng,sim_sigma0,' ; '.join(hyperprior_lines)),fontsize=plotter.FS)
fig.axes[-1].set_xlabel(r'%s'%parlabels_full[loop_par],fontsize=plotter.FS)
fig.axes[0].set_ylabel('Posterior Densities',fontsize=plotter.FS,color='white')#For spacing
fig.text(0.01, 0.5, 'Posterior Densities', ha='center', va='center', rotation='vertical',fontsize=plotter.FS)
pl.tight_layout()
if args['save']:#savekey   = f'multigalsims_{samplename}_Modelsigma0{sigma0}'
	pl.savefig(f"{plotpath}SBC_looppar{loop_par}_quantilemode{args['quantilemode']}_hyperprior{ALT_PRIOR}.pdf",bbox_inches='tight')
if args['show']:
	pl.show()


#Plot of simulation averaged posteriors of pars listed below
pars = ['rho','sigmaRel','sigmaCommon','sigma0']
pars = ['rho','rel_rat','com_rat']
GFITS = {key:value for key,value in GLOB_FITS.items() if key in [1,0]}
fig,axs = pl.subplots(len(GFITS),len(pars),figsize=(8*len(pars),6*len(GFITS)),sharex=False)
counter=-1
for row,true_rho in enumerate(GFITS):
	for col,par in enumerate(pars):
		counter+=1
		FITS      = GFITS[true_rho]
		sap,Ncred = get_sap_Ncred(par,true_rho,ALT_PRIOR)
		true_par  = round(df_vals[df_vals.rho==true_rho][par].values[0],3)
		plotter   = SBC_FITS_PLOTTER(counter,fig.axes,[true_par,par,dfpars[par],parlabels[par]],FITS,bounds,rootpath,quantilemode=args['quantilemode'])
		plotter.plot_sbc_panel(sap=sap,Ncred=False,Lside=True if sap==1 else False,
										annotate_true=True,plot_ind=False,plot_true=True,plot_medians=False,dress_figure=True,
										fill_between=False,color=f"C0",linestyle='-',FAC=100,line_sap_title='')#'Sim. Posterior')
		if row==len(GFITS)-1:
			fig.axes[counter].set_xlabel(r'%s'%parlabels_full[par],fontsize=plotter.FS+2)
for iax in range(int(len(GFITS)*len(pars))):
	new_xlabels = [x if float(x.get_text()) not in [0,0.1,1] else {0:'0',0.1:'0.1',1:'1'}[float(x.get_text())] for x in fig.axes[iax].get_xticklabels()]
	fig.axes[iax].set_xticklabels(new_xlabels)
	pass

fig.axes[0].set_ylabel('Posterior Densities',fontsize=plotter.FS+2,color='white')#For spacing
fig.text(0.1, 0.5, 'Posterior Densities', ha='center', va='center', rotation='vertical',fontsize=plotter.FS)
fig.subplots_adjust(wspace=0.08,hspace=0.08)
if args['save']:
	print (plotpath)
	pl.savefig(f"{plotpath}SBC_{'.'.join(pars)}.pdf",bbox_inches='tight')
if args['show']:
	pl.show()
