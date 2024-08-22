import argparse,sys,yaml
rootpath = '../'
sys.path.append(rootpath+'model_files/')
from sbc_plot_functions import *
from sigmaRel import *
from simulate_distances import *

#SBC settings
Nsim_keep = 100
Ng = 100 ; Sg = 2; RS=100 ; sigma0 = 'free' ; sigma_fit_s = 0.05
sigma_fit_s = 0.1
RHOS = [1,0]

pars = ['rho','sigmaRel','sigmaCommon','sigma0'][1:-1]
pars = ['rho','rel_rat','com_rat']

#Initialise args
parser = argparse.ArgumentParser(description="SBC Input Hyperparameters")
parser.add_argument("--save",	default=True,		      help='Save plot')
parser.add_argument("--show",	default=False,		      help='Show plot')
parser.add_argument("--quantilemode",	default=True,	  help='If True, annotate with 16,84, False, use sample std.')
args = parser.parse_args().__dict__

#Perform SBC
ALT_PRIORS = [False,'C']
Summary_Strs = []; GLOB_FITS = {} ; productpath = None; plotpath = None; modelkey  = None; sim_sigma0 = None
simulator  = SiblingsDistanceSimulator(Ng=Ng,Sg=Sg,external_distances=True,sigmaRel=0.1,zcmberr=1e-5,random=42,sigma_fit_s=sigma_fit_s)
dfmus      = simulator.dfmus
dfmus['zhelio_hats'] = dfmus['zcmb_hats'] ; dfmus['zhelio_errs'] = dfmus['zcmb_errs']

modelkeys = {}
for altprior in ALT_PRIORS:
	multigal = multi_galaxy_siblings(dfmus,rootpath=rootpath)
	multigal.sigmaRel_sampler(alt_prior=altprior,sigma0=sigma0,use_external_distances=True,zcosmo='zcmb',overwrite=False)
	modelkeys[altprior] = copy.deepcopy(multigal.modelkey)
	if sim_sigma0 is None:
		productpath = copy.deepcopy(multigal.productpath)
		plotpath    = copy.deepcopy(multigal.plotpath)+'sbc/'
		if not os.path.exists(plotpath):    os.mkdir(plotpath)
		sim_sigma0   = copy.deepcopy(simulator.sigma0)
		sim_sigmafit = copy.deepcopy(simulator.sigma_fit_s[0])
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
ALL_FITS = {}
for altprior in ALT_PRIORS:
	GLOB_FITS = {}
	for sigR,RHO in zip(df_vals.sigmaRel.values,df_vals.rho.values):
		FITS = {}
		for ISIM in np.arange(RS):
			samplename = f'Ng{Ng}_Sg{Sg}_Rs{ISIM}_Truesigma0{sim_sigma0}_TruesigmaRel{0 if sigR==0 else round(sigR,3)}'+(sim_sigmafit!=0.05)*f'_sigmafit{sim_sigmafit}'
			filename   = productpath+f"FIT{samplename}{modelkeys[altprior]}.pkl"
			with open(filename,'rb') as f:
				FIT = pickle.load(f)
			chains, summary = multigal.add_transformed_params(FIT['chains'], FIT['summary'])#Add transformed parameters
			FIT['chains'] = chains; FIT['summary'] = summary
			FITS[ISIM] = FIT
		GLOB_FITS[RHO] = FITS
	ALL_FITS[altprior] = GLOB_FITS

#Plot SBC
fig,axs = pl.subplots(len(RHOS),len(pars),figsize=(8*len(pars),6*len(RHOS)),sharex=False)
for ip,altprior in enumerate(ALL_FITS):
	GFITS = {key:value for key,value in ALL_FITS[altprior].items() if key in RHOS}
	counter=-1
	for row,true_rho in enumerate(GFITS):
		for col,par in enumerate(pars):
			counter+=1
			FITS      = GFITS[true_rho]
			sap,Ncred = get_sap_Ncred(par,true_rho,altprior)
			true_par  = round(df_vals[df_vals.rho==true_rho][par].values[0],3)
			plotter   = SBC_FITS_PLOTTER(counter,fig.axes,[true_par,par,dfpars[par],parlabels[par]],FITS,bounds,rootpath,quantilemode=args['quantilemode'])
			plotter.plot_sbc_panel(sap=sap,Ncred=False,Lside=True if sap==1 else False,
											#Parcred=True,
											annotate_true=True,plot_ind=False,plot_true=True,plot_medians=False,dress_figure=False,
											fill_between=False,color=f"C{ip}",linestyle='-',FAC=20,line_sap_title='')
			if row==len(GFITS)-1:
				fig.axes[counter].set_xlabel(r'%s'%parlabels_full[par],fontsize=plotter.FS+2)

for iax in range(int(len(RHOS)*len(pars))):
	fig.axes[iax].set_yticks([])
	fig.axes[iax].legend(fontsize=plotter.FS,framealpha=1,loc='upper right')
	fig.axes[iax].tick_params(labelsize=plotter.FS)
	y = fig.axes[iax].get_ylim()
	fig.axes[iax].set_ylim([0,y[1]])
	new_xlabels = [x if float(x.get_text()) not in [0,0.1,1] else {0:'0',0.1:'0.1',1:'1'}[float(x.get_text())] for x in fig.axes[iax].get_xticklabels()]
	fig.axes[iax].set_xticklabels(new_xlabels)

fig.axes[0].set_title('NA\nNA\nNA',fontsize=plotter.FS+4,color='white')#For spacing
fig.text(0.5,0.99,'Simulation-Based Calibration; Hyperpriors:',fontsize=plotter.FS+4,ha='center', va='center',color='black',weight='bold')#For spacing
fig.text(0.5,0.95,' ; '.join([r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim \rm{Arcsine}(0,1)$"]),fontsize=plotter.FS+4,ha='center', va='center',color='C0',weight='bold')#For spacing
fig.text(0.5,0.91,' ; '.join([r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim U(0,1)$"]),fontsize=plotter.FS+4,ha='center', va='center',color='C1',weight='bold')#For spacing
fig.axes[0].set_ylabel('Simulation-Averaged Posteriors',fontsize=plotter.FS+2,color='white')#For spacing
fig.text(0.1, 0.5, 'Simulation-Averaged Posteriors', ha='center', va='center', rotation='vertical',fontsize=plotter.FS)
fig.subplots_adjust(wspace=0.08,hspace=0.08)
if args['save']:
	pl.savefig(f"{plotpath}SBC_{'.'.join(pars)}.pdf",bbox_inches='tight')
if args['show']:
	pl.show()
