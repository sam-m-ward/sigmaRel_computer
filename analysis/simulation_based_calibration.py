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
Ng = 100 ; Sg = 2 ; RS=100
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
Summary_Strs = []; GLOB_FITS = {} ; productpath = None; plotpath = None; modelkey  = None; sim_sigma0 = None
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

			if modelkey is None:
				productpath = copy.deepcopy(multigal.productpath)
				plotpath    = copy.deepcopy(multigal.plotpath)+'sbc/'
				if not os.path.exists(plotpath):    os.mkdir(plotpath)
				sim_sigma0  = copy.deepcopy(simulator.sigma0)
				modelkey    = copy.deepcopy(multigal.modelkey)
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
for sigR,RHO in zip(df_vals.sigmaRel.values,df_vals.rho.values):
	FITS = {}
	for ISIM in np.arange(RS):
		samplename = f'Ng{Ng}_Sg{Sg}_Rs{ISIM}_Truesigma0{sim_sigma0}_TruesigmaRel{0 if sigR==0 else round(sigR,3)}'
		filename   = productpath+f"FIT{samplename}{modelkey}.pkl"
		with open(filename,'rb') as f:
			FIT = pickle.load(f)
		chains, summary = multigal.add_transformed_params(FIT['chains'], FIT['summary'])#Add transformed parameters
		FIT['chains'] = chains; FIT['summary'] = summary
		FITS[ISIM] = FIT
	FITS = trim_to_KEEPERS({'dummy':FITS},get_KEEPERS({'dummy':FITS},Nsim_keep,Rhat_threshold,loop_par,dfpars[loop_par]))['dummy']
	GLOB_FITS[RHO] = FITS

#Plot SBC
fig,axs = pl.subplots(len(GLOB_FITS),1,figsize=(8,6*len(GLOB_FITS)),sharex=False)
for iax,true_rho in enumerate(GLOB_FITS):
	FITS     = GLOB_FITS[true_rho]
	###GET SAP KEY##############################################
	sap = None
	if loop_par=='rho':
		sap      = true_rho if true_rho in [0,1] else None
	if (ALT_PRIOR is False) and ((true_rho==1 and loop_par in ['sigmaRel','rel_rat','rel_rat2']) or (true_rho==0 and loop_par in ['sigmaCommon','com_rat','com_rat2'])):
			sap=0
	if (true_rho==0 and loop_par in ['rel_rat','rel_rat2']) or (true_rho==1 and loop_par in ['com_rat','com_rat2']):
			sap=1
	###GET NCRED
	Ncred = False if (sap is None and true_rho!=0.5 and (  (loop_par not in ['sigma0','sigmaRel','sigmaCommon'] and ALT_PRIOR is False) \
														or (loop_par not in ['sigma0'] and ALT_PRIOR is not False)))  else True
	if ALT_PRIOR is not False:
		if (true_rho==0 and loop_par in ['sigmaRel']) or (true_rho==1 and loop_par in ['sigmaCommon']):
			Ncred = True
	############################################################
	true_par = round(df_vals[df_vals.rho==true_rho][loop_par].values[0],3)
	plotter  = SBC_FITS_PLOTTER(iax,fig.axes,[true_par,loop_par,dfpars[loop_par],parlabels[loop_par]],FITS,bounds,rootpath,quantilemode=args['quantilemode'])
	plotter.plot_sbc_panel(sap=sap, Lside = True if ('_rat' in loop_par and true_rho==0.5) else False,
									Ncred = Ncred,FAC = 25)
fig.axes[0].set_title('Fits to %s$\\times$%s simulations of %s sibling-pair galaxies;\nTrue $\sigma_0 = %s\,$mag;\nHyperpriors: %s'%(df_vals.shape[0],Nsim_keep,Ng,sim_sigma0,' ; '.join(hyperprior_lines)),fontsize=plotter.FS)
fig.axes[-1].set_xlabel(r'%s'%parlabels_full[loop_par],fontsize=plotter.FS)
fig.axes[0].set_ylabel('Posterior Densities',fontsize=plotter.FS,color='white')#For spacing
fig.text(0.01, 0.5, 'Posterior Densities', ha='center', va='center', rotation='vertical',fontsize=plotter.FS)
pl.tight_layout()
if args['save']:
	#savekey   = f'multigalsims_{samplename}_Modelsigma0{sigma0}'
	pl.savefig(f"{plotpath}SBC_looppar{loop_par}_quantilemode{args['quantilemode']}_hyperprior{ALT_PRIOR}.pdf",bbox_inches='tight')
if args['show']:
	pl.show()
