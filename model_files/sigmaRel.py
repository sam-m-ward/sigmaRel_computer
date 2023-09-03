"""
Sigma_Rel Model

Module containing siblings_galaxy class
Operations for plotting individual siblings distances, computing sigma-rel posteriors, and combining distance estimates


Contains:
--------------------
multi_galaxy class:
	inputs: dfmus,samplename='multigal',sigma0=0.1,prior_upper_bounds=[1.0],rootpath='./'

	Methods are:
		compute_analytic_multi_gal_sigmaRel_posterior()
		loop_single_galaxy_analyses()

siblings_galaxy class:
	inputs: mus,errors,names,galname,prior_upper_bounds=[0.1,0.15,1.0],sigma0=0.094,Ngrid=1000,fontsize=18,show=False,save=True

	Methods are:
		get_sigmaRel_posterior(prior_distribution='uniform', prior_upper_bound = 1.0)
		get_sigmaRel_posteriors()
		get_CDF()
		get_quantile(q)
		get_weighted_mu()
		combine_individual_distances(mode=None,overwrite=True)
		plot_individual_distances(colours=None,markers=None,markersize=10,capsize=8,mini_d=0.025,plot_full_errors=True)
		plot_sigmaRel_posteriors(xupperlim=0.16,colours = ['green','purple','goldenrod'])
		plot_common_distances(markersize=10,capsize=8,mini_d=0.025)
--------------------

Written by Sam M. Ward: smw92@cam.ac.uk
"""

import arviz as az
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as pl
import numpy as np
import copy, os, pickle, re

class multi_galaxy_siblings:

	def __init__(self,dfmus,samplename='multigal',sigma0=0.1,prior_upper_bounds=[1.0],rootpath='./'):
		"""
		Initialisation

		Parameters
		----------
		dfmus : pandas df
			dataframe of individual siblings distance estimates with columns Galaxy, SN, mus, mu_errs

		samplename : str (optional; default='multigal')
			name of multi-galaxy sample of siblings

		sigma0 : float (optional; default=0.094~mag i.e. the W22 training value)
			the total intrinsic scatter, i.e. informative prior upper bound on sigmaRel

		prior_upper_bounds : list (optional; default=[1.0])
			choices of sigmaRel prior upper bound

		rootpath : str
			path/to/sigmaRel/rootpath
		"""
		self.dfmus      = dfmus
		self.samplename = samplename
		self.sigma0     = sigma0
		self.prior_upper_bounds = prior_upper_bounds
		self.rootpath   = rootpath

		self.modelpath   = self.rootpath  + 'model_files/'
		self.stanpath    = self.modelpath + 'stan_files/MultiGalFiles/'
		self.productpath = self.rootpath  + 'products/multigal/'
		self.plotpath    = self.rootpath  + 'plots/multi_galaxy_plots/'

		self.n_warmup   = 1000
		self.n_sampling = 5000
		self.n_chains   = 4

	def sigmaRel_sampler(self, sigma0=None, sigmapec=None, use_external_distances=False,eta_sigmaRel_input=None,overwrite=True):
		"""

		"""
		c_light = 299792458
		if sigma0 is None:			sigma0   = self.sigma0	#If not specified, used the class input value (which if itself is not specified has a default value sigma0=0.1)
		if sigmapec is None: 		sigmapec = 250			#If not specified, fix sigmapec
		if eta_sigmaRel_input is None:
			sigmaRel_input     = 0  #If not specified, free sigmaRel
			eta_sigmaRel_input = 0
		else:#Otherwise, fix sigmaRel to fraction of sigma0
			assert(type(eta_sigmaRel_input) in [float,int])
			assert(0<=eta_sigmaRel_input and eta_sigmaRel_input<=1)
			sigmaRel_input     = 1

		#Stan HBM files for the different intrinsic scatter modelling assumptions
		sigmas   = {'sigma0':{'value':sigma0},'sigmapec':{'value':sigmapec}}
		muextstr = {False:'no_muext',True:'with_muext'}[use_external_distances]
		for label in sigmas.keys():
			value = sigmas[label]['value']
			if type(value) in [float,int] and value>0:
				sigmastr = f"fixed_{label}{value}"
				if (use_external_distances and label=='sigmapec') or label in ['sigma0']:
					print (f"{label} fixed at {value}")
			elif value=='free':
				sigmastr = f"free_{label}"
				print (f"{label} is a free hyperparameter")
			else:
				raise Exception(f"{label} must be float/int >0 or 'free', but {label}={value}")
			sigmas[label]['str'] = sigmastr

		#Model being used
		modelkey = f"{sigmas['sigma0']['str']}_{muextstr}"
		if sigmas['sigma0']['value']!='free' and not use_external_distances:
			print('No external distances used')
		elif sigmas['sigma0']['value']=='free' and not use_external_distances:
			raise Exception('Freeing sigma0 without external distances')
		else:
			print ('Using external distances')
			modelkey += f"_{sigmas['sigmapec']['str']}"
		if bool(sigmaRel_input):
			print (f"sigmaRel is fixed at {eta_sigmaRel_input}*sigma0")
			modelkey += f"_etasigmaRelfixed{eta_sigmaRel_input}"
		else:
			print (f"sigmaRel is free hyperparameter")
			modelkey += f"_sigmaRelfree"

		"""
		 POTENTIAL stan_files		:	 MODELKEYS
		'sigmaRel_nomuext.stan'		:	'fixed_sigma0_no_muext'
		'sigmaRe_withmuext.stan'	:	'fixed_sigma0_with_muext_fixed_sigmapec'
										'fixed_sigma0_with_muext_free_sigmapec'
										'free_sigma0_with_muext_fixed_sigmapec'
										'free_sigma0_with_muext_free_sigmapec'
		For each we can have sigmaRelfree OR etasigmaRelfixed
		"""
		stan_file = {True:'sigmaRel_withmuext.stan',False:'sigmaRel_nomuext.stan'}[use_external_distances]
		with open(self.stanpath+stan_file,'r') as f:
			stan_file = f.read().splitlines()
		if ('fixed_sigmapec' in modelkey or 'fixed_sigma0' in modelkey) and use_external_distances:
			for il, line in enumerate(stan_file):
				if 'fixed_sigmapec' in modelkey:
					if bool(re.match(r'\s*//real<lower=0,upper=1>\s*pec_unity;\s*//Data',line)):
						stan_file[il] = line.replace('//real','real')
					if bool(re.match(r'\s*real<lower=0,upper=1>\s*pec_unity;\s*//Model',line)):
						stan_file[il] = line.replace('real','//real')
					if bool(re.match(r'\s*pec_unity\s*~\s*uniform\(0,1\)',line)):
						stan_file[il] = line.replace('pec_unity','//pec_unity')
				if 'fixed_sigma0' in modelkey:
					if bool(re.match(r'\s*//real<lower=0,upper=1>\s*sigma0;\s*//Data',line)):
						stan_file[il] = line.replace('//real','real')
					if bool(re.match(r'\s*real<lower=0,upper=1>\s*sigma0;\s*//Model',line)):
						stan_file[il] = line.replace('real','//real')
					if bool(re.match(r'\s*sigma0\s*~\s*uniform\(0,1\)',line)):
						stan_file[il] = line.replace('sigma0','//sigma0')
		stan_file = '\n'.join(stan_file)
		print (stan_file)
		with open(self.stanpath+'current_model.stan','w') as f:
			f.write(stan_file)


		#If files don't exist or overwrite, do stan fits
		if not os.path.exists(self.productpath+f"FITS{self.samplename}{modelkey}.pkl") or overwrite:
			#Pars of interest
			pars = ['sigmaRel','sigma0','sigmapec']
			#For each mode, perform stan fit to combine distances
			print ('###'*30)
			print (f"Beginning Stan fit: {modelkey}")
			dfmus = copy.deepcopy(self.dfmus)
			#Groupby galaxy to count Nsibs per galaxy
			Gal   = dfmus.groupby('Galaxy',sort=False)[['muext_hats','zcmb_hats','zcmb_errs']].agg('mean')
			Ng    = Gal.shape[0]
			S_g   = dfmus.groupby('Galaxy',sort=False)['SN'].agg('count').values
			Nsibs = int(sum(S_g))
			#Individual siblings distance estimates
			mu_sib_phots     = dfmus['mus'].values
			mu_sib_phot_errs = dfmus['mu_errs'].values
			#External Distances
			mu_ext_gal, zcmbs, zcmberrs = [Gal[col].tolist() for col in Gal.columns]

			#Load up data
			stan_data = dict(zip(['Ng','S_g','Nsibs','mu_sib_phots','mu_sib_phot_errs','mu_ext_gal','zcmbs','zcmberrs','sigmaRel_input','eta_sigmaRel_input'],
								 [ Ng , S_g , Nsibs , mu_sib_phots , mu_sib_phot_errs , mu_ext_gal , zcmbs , zcmberrs , sigmaRel_input , eta_sigmaRel_input ]))
			if not use_external_distances:
				stan_data = {key:value for key,value in stan_data.items() if key not in ['mu_ext_gal','zcmbs','zcmberrs']}
			if sigma0!='free':
				stan_data['sigma0']    = sigma0
				pars.remove('sigma0')
			if sigmapec!='free':
				stan_data['pec_unity'] = sigmapec*1e3/c_light
				pars.remove('sigmapec')

			model = CmdStanModel(stan_file=self.stanpath+'current_model.stan')

			fit         = model.sample(data=stan_data,chains=self.n_chains, iter_sampling=self.n_sampling, iter_warmup = self.n_warmup)
			fitsummary  = az.summary(fit)
			df          = fit.draws_pd()

			for col in pars:
				print ('###'*10)
				print (f'{col}, median, std, 16, 84 // 68%, 95%')
				print (fitsummary.loc[col]['r_hat'])
				print (df[col].median().round(3),df[col].std().round(3),df[col].quantile(0.16).round(3),df[col].quantile(0.84).round(3))
				print (df[col].quantile(0.68).round(3),df[col].quantile(0.95).round(3))
			#STORE = {'median':round(df['mu'].median(),3),'std':round(df['mu'].std(),3)}
			#print (f"Fit Completed; Summary is:")
			#print (fitsummary.loc[f'mu'])
			#print ('Estimates of distance')
			#print ('~~~'*30)
			#print (f"dM-{mode.capitalize()} Common-mu = {round(df['mu'].median(),3)}+/-{round(df['mu'].std(),3)}")
			#print ('~~~'*30)
			#print ('###'*30)
			err=1/0
			#FIT = dict(zip(['data','summary','chains'],[stan_data,fitsummary,df]))
			#with open(self.productpath+f"FITS{self.galname}.pkl",'wb') as f:
			#	pickle.dump({'FIT':FIT,'common_distance_estimates':STORE},f)
		else:#Else load up
			with open(self.productpath+f"FITS{self.galname}.pkl",'rb') as f:
				loader = pickle.load(f)
			STORE = loader['common_distance_estimates']
			FIT   = loader['FITS']

		#Print distance summaries
		print (STORE)
		for x in STORE:
			print (x,STORE[x])

		#Save data, summaries and chains
		self.common_distance_estimates = STORE
		self.FITS  = FITS



	def compute_analytic_multi_gal_sigmaRel_posterior(self):
		"""
		Compute Analytic Multi Galaxy sigmaRel Posterior

		Method to compute analytic sigmaRel posterior by multiplying the single-galaxy likelihoods by the prior

		End Product(s)
		----------
		Plot of multi-galaxy sigmaRel posterior

		self.total_posteriors: dict
			key,value pairs are the sigmaRel prior upper bound, and the posterior

		self.sigRs_store: dict
		 	same as self.total_posteriors, but the sigR prior grid
		"""
		#Initialise posteriors
		total_posteriors = {p:1 for p in self.prior_upper_bounds} ; sigRs_store = {}
		#For each galaxy, compute sigmaRel likelihood
		for g,gal in enumerate(self.dfmus['Galaxy'].unique()):
			dfgal  = self.dfmus[self.dfmus['Galaxy']==gal]
			sibgal = siblings_galaxy(dfgal['mus'].values,dfgal['mu_errs'].values,dfgal['SN'].values,gal,sigma0=self.sigma0,prior_upper_bounds=self.prior_upper_bounds)
			sibgal.get_sigmaRel_posteriors()
			for p in self.prior_upper_bounds:
				total_posteriors[p] *= sibgal.posteriors[p]*p #Multiply likelihoods, so divide out prior for each galaxy
				if g==0:
					total_posteriors[p] *= 1/p#Prior only appears once
					sigRs_store[p] = sibgal.sigRs_store[p]

		#Use single-galaxy class for plotting
		for p in self.prior_upper_bounds:
			sibgal.posteriors[p]  = total_posteriors[p]
			sibgal.sigRs_store[p] = sigRs_store[p]

		#Plot posteriors
		sibgal.galname = self.samplename
		sibgal.plotpath = self.plotpath
		sibgal.plot_sigmaRel_posteriors()

		#Store posteriors as attribute
		self.total_posteriors = total_posteriors
		self.sigRs_store      = sigRs_store


	def loop_single_galaxy_analyses(self):
		"""
		Loop Single Galaxy Analyses

		Method to take pandas df of multi-galaxy siblings distances, and do Ngal single-galaxy analyses

		End Product(s)
		----------
		x3 plots per galaxy: Individual Distance Estimates, sigmaRel posteriors, Common-mu posteriors
		"""
		for gal in self.dfmus['Galaxy'].unique():
			dfgal  = self.dfmus[self.dfmus['Galaxy']==gal]
			sibgal = siblings_galaxy(dfgal['mus'].values,dfgal['mu_errs'].values,dfgal['SN'].values,gal,sigma0=self.sigma0)
			sibgal.plot_individual_distances()
			sibgal.plot_sigmaRel_posteriors()
			sibgal.combine_individual_distances()
			sibgal.plot_common_distances()



class siblings_galaxy:

	def __init__(self,mus,errors,names,galname,prior_upper_bounds=[0.1,0.15,1.0],sigma0=0.094,Ngrid=1000,rootpath='./',fontsize=18,show=False,save=True):
		"""
		Initialisation

		Parameters
		----------
		mus : array
			siblings distance estimates to a single galaxy

		errors : array
			the measurement errors, or BayeSN 'fitting uncertainties', on each individual distance estimate

		names : lst
			names of each SN sibling

		galname : str
			name of siblings galaxy

		prior_upper_bounds : lst (optional; default=[0.1,0.15,1.0])
			choices of sigmaRel prior upper bound

		sigma0 : float (optional; default=0.094~mag i.e. the W22 training value)
			the total intrinsic scatter, i.e. informative prior upper bound on sigmaRel

		Ngrid : int (optional; default=1000)
			the number of sigmaRel grid points used in prior for computing posterior in range[0,prior_upper_bound]

		rootpath : str
			path/to/sigmaRel/rootpath

		fontsize : float (optional; default=18)
			fontsize for plotting

		show : bool (optional; default=False)
			bool to show plots or not

		save : bool (optional; default=True)
			bool to save plots or not
		"""
		self.mus     = mus
		self.errors  = errors
		self.names   = names
		self.galname = galname
		self.prior_upper_bounds = prior_upper_bounds
		self.sigma0   = sigma0
		self.Ngrid    = Ngrid
		self.rootpath = rootpath
		self.FS       = fontsize
		self.save     = save
		self.show     = show

		self.Sg         = int(len(mus))
		self.fullerrors = np.array([self.sigma0**2 + err**2 for err in self.errors])**0.5

		self.n_warmup   = 250
		self.n_sampling = 25000
		self.n_chains   = 4

		self.modelpath   = self.rootpath  + 'model_files/'
		self.stanpath    = self.modelpath + 'stan_files/'
		self.productpath = self.rootpath  + 'products/'
		self.plotpath    = self.rootpath  + 'plots/single_galaxy_plots/'

	def get_sigmaRel_posterior(self, prior_distribution='uniform', prior_upper_bound = 1.0):
		"""
		Get Sigma_Rel Posterior

		Method to compute sigmaRel posterior given individual distances and prior upper bound

		Parameters
		----------
		prior_distribution : str (optional; default='uniform')
			denotes the form of the prior distribution

		prior_upper_bound : float (optional; default=1.0)
			float value of prior upper bound on sigmaRel posterior

		End Product(s)
		----------
		self.sigRs,self.posterior : arrays
			the prior grid, and the analytic posterior
		"""
		def Gaussian(x,mu,sig):
			num = np.exp( -0.5*( ((x-mu)/sig)**2 ) )
			den = (2*np.pi*sig**2)**0.5
			return num/den

		#Check prior distribution is correct
		if prior_distribution!='uniform': raise Exception('Not yet implemented other prior distributions')

		#Get prior grid and prior normalisation
		self.sigRs      = np.linspace(0, prior_upper_bound, self.Ngrid)
		self.prior_sigR = 1/prior_upper_bound

		#Compute posterior
		posterior_values = []
		for sgR in self.sigRs:
			vars    = np.array([sgR**2+error**2 for error in self.errors])
			weights = 1/vars
			weighted_mean_mu = sum(weights*self.mus)/sum(weights)
			posterior = self.prior_sigR/(sum(weights)**0.5)
			for mu,sigmafit,var in zip(self.mus,self.errors,vars):
				posterior *= Gaussian(mu,weighted_mean_mu,var**0.5)
			posterior_values.append(posterior)
		self.posterior = np.asarray(posterior_values)

	def get_sigmaRel_posteriors(self):
		"""
		Get Sigma_Rel Posteriors

		Method to loop over choices of prior upper bound and get individual sigmaRel posteriors

		End Products
		----------
		self.posteriors : dict
			key,value pairs are the prior upper bound, and the analytic posterior
		self.sigRs_store : dict
			same as self.posteriors, but value is grid of prior sigRs
		"""
		self.posteriors  = {} ; self.sigRs_store = {}
		for prior_upper_bound in self.prior_upper_bounds:
			self.get_sigmaRel_posterior(prior_upper_bound=prior_upper_bound)
			self.posteriors[prior_upper_bound]  = self.posterior
			self.sigRs_store[prior_upper_bound] = self.sigRs

	def get_CDF(self):
		"""
		Get CDF

		Returns cumulative distribution function of un-normalised sigmaRel posterior

		End Product
		----------
		self.CDF : array
			cumulative sum of un-normalised posterior
		"""
		self.CDF = np.array([sum(self.posterior[:i])/sum(self.posterior) for i in range(len(self.posterior))])

	def get_quantile(self, q):
		"""
		Get Quantile

		Returns posterior quantiles using the CDF

		Parameters
		----------
		q : float
			percentage value between 0 and 1 for quantile

		Returns
		----------
		sigR_q : float
			the sigR grid-value at the posterior quantile, q

		index_q : int
			the index on the prior grid for that quantile
		"""
		self.get_CDF()
		index_q = np.argmin(np.abs(self.CDF-q))
		sigR_q  = self.sigRs[index_q]
		return sigR_q, index_q

	def get_weighted_mu(self):
		"""
		Get weighted mu

		Method for determining precision-weighted average distance estimate
		This is equivalent to adopting the dM-Uncorrelated assumption
		The errors include sigma0 (excluding this is equivalent to dM-Common assumption, in which case sigma0 should then be added in quadrature to uncertainty)

		End Product(s)
		----------
		self.weighted_mu : float
			the precision-weighted average distance estimate

		self.err_weighted_mu : floatt
			the uncertainty on the precision-weighted average distance estimate
		"""
		weights = 1/(self.fullerrors**2)
		self.weighted_mu     = sum(self.mus*weights)/sum(weights)
		self.err_weighted_mu = (1/sum(weights))**0.5

	def combine_individual_distances(self,mode=None,overwrite=True):
		"""
		Combine Individual Distances

		Method to take individual siblings distance estimates, and combine them according to a choice of intrinsic modelling assumptions
		By default, method runs through all x3 modelling assumptions (Perfectly Uncorrelated, Fit for Correlation, Perfectly Correlated)

		Parameters
		----------
		mode : str (optional; default is None)
			choice to define a specific intrinsic scatter modelling assumption, otherwise loops through all 3

		overwrite : bool (optional; default=True)
			if True, re-run stan fits

		End Product(s)
		----------
		self.common_distance_estimates : dict
			{key,value} are the mode and x
			where x is median and std of common distance estimate in dict form

		self.STORE : dict
			keys are modes, values.keys() are ['data','summary','chains'] from posterior fits

		"""
		#Stan HBM files for the different intrinsic scatter modelling assumptions
		stan_files = {  'uncorrelated'  :'CombineIndividualDistances_dMUncorrelated.stan',
						'mixed'         :'CombineIndividualDistances_dMMixed.stan',
						'common'        :'CombineIndividualDistances_dMCommon.stan'}

		#Get modes to loop over
		modes = list(stan_files.keys())
		if mode is not None:
			if mode not in modes:	raise Exception(f"User-inputted mode={mode} is not in {modes}")
			else:					modes = [mode]

		#If files don't exist or overwrite, do stan fits
		if not os.path.exists(self.productpath+f"FITS{self.galname}.pkl") or overwrite:
			#For each mode, perform stan fit to combine distances
			STORE = {} ; FITS = {}
			print ('###'*30)
			for mode in modes:
				print (f"Beginning Stan fit adopting the dM-{mode.capitalize()} assumption")
				stan_data = dict(zip(['S','sigma0','mean_mu','mu_s','mu_err_s'],[self.Sg,self.sigma0,np.average(self.mus),self.mus,self.errors]))
				model = CmdStanModel(stan_file=self.stanpath+stan_files[mode])
				fit         = model.sample(data=stan_data,chains=self.n_chains, iter_sampling=self.n_sampling, iter_warmup = self.n_warmup)
				fitsummary  = az.summary(fit)
				df          = fit.draws_pd()
				STORE[mode] = {'median':round(df['mu'].median(),3),'std':round(df['mu'].std(),3)}
				print (f"Fit Completed; Summary is:")
				print (fitsummary.loc[f'mu'])
				print ('Estimates of distance')
				print ('~~~'*30)
				print (f"dM-{mode.capitalize()} Common-mu = {round(df['mu'].median(),3)}+/-{round(df['mu'].std(),3)}")
				print ('~~~'*30)
				print ('###'*30)
				FITS[mode] = dict(zip(['data','summary','chains'],[stan_data,fitsummary,df]))
			with open(self.productpath+f"FITS{self.galname}.pkl",'wb') as f:
				pickle.dump({'FITS':FITS,'common_distance_estimates':STORE},f)
		else:#Else load up
			with open(self.productpath+f"FITS{self.galname}.pkl",'rb') as f:
				loader = pickle.load(f)
			STORE = loader['common_distance_estimates']
			FITS  = loader['FITS']

		#Print distance summaries
		print (STORE)
		for x in STORE:
			print (x,STORE[x])

		#Save data, summaries and chains
		self.common_distance_estimates = STORE
		self.FITS  = FITS

	def plot_individual_distances(self,colours=None,markers=None,markersize=10,capsize=8,mini_d=0.025,plot_full_errors=True):
		"""
		Plot Individual Distances

		Method to plot up the individual distance estimates

		Parameters
		----------
		colours : lst (optional; default is None)
			colours for each distance estimate

		markers : lst (optional; default is None)
			markers for each distance estimate

		markersize : float (optional; default=10)
			size of distance scatter point markers

		capsize : float (optional; default=8)
			size of errorbar caps

		mini_d : float (optionl; default=0.025)
			delta_x value for separating distances visually

		plot_full_errors : bool (optional; default=True)
			plot the full errors which include the sigma0 contribution

		End Product
		----------
		Plot of individual distance estimates
		"""
		fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
		fig.axes[0].set_title(r"Individual Siblings Distance Estimates",weight='bold',fontsize=self.FS+1)

		if colours is None: colours = [f'C{s}' for s in range(self.Sg)]
		if markers is None: markers = [m for m in ['o','s','p','^','x','P','d','*'][:self.Sg]]

		deltas = [mini_d*(ss-(self.Sg-1)/2) for ss in range(self.Sg)]
		for mu,err,fullerr,ss in zip(self.mus,self.errors,self.fullerrors,np.arange(self.Sg)):
			fig.axes[0].errorbar(deltas[ss],mu,yerr=err,     color=colours[ss],marker=markers[ss],markersize=markersize,          linestyle='none',capsize=capsize, label=self.names[ss])
			if plot_full_errors:
				fig.axes[0].errorbar(deltas[ss],mu,yerr=fullerr, color=colours[ss],marker=markers[ss],markersize=markersize,alpha=0.4,linestyle='none',capsize=capsize)
			fig.axes[0].set_xticklabels("")

		self.get_weighted_mu()
		Ylow  = min(self.mus)-self.fullerrors[np.argmin(self.mus)]*1.5 - 0.05
		DY    = 0.03 ; dx = mini_d*0.2
		fig.axes[0].annotate(r'Weighted-Average $\mu$:',		ha='left',xy=(deltas[0]-mini_d*0.6+dx,Ylow+DY),color='black',fontsize=self.FS)
		fig.axes[0].annotate(f'{round(self.weighted_mu,3)}'+r'$\pm$'+f'{round(self.err_weighted_mu,3)}' + ' mag',ha='right',xy=(deltas[-1]+dx*1.5,Ylow+DY),color='black',fontsize=self.FS)
		fig.axes[0].annotate(r'Std. Dev. of $\mu$ Point Estimates:',	ha='left',xy=(deltas[0]-mini_d*0.6+dx,Ylow),    color='black',fontsize=self.FS)
		fig.axes[0].annotate(str(round(np.std(self.mus),3)) + ' mag',											ha='right',xy=(deltas[-1]+dx*1.5,Ylow),color='black',fontsize=self.FS)

		fig.axes[0].annotate(#f"NGC 3147's Siblings"+'\n'+r'$BayeSN$ Distance Estimates'+'\n'+r'$R_V^s \sim U(1,6)$',
							 f"{self.galname}'s Siblings",#+'\n'+r'$BayeSN$ Distance Estimates'+'\n'+r'$R_V^s \sim U(1,6)$',
								xy = (deltas[0]-mini_d*0.6+dx,Ylow+2*DY),ha='left',
								fontsize=self.FS)
		fig.axes[0].set_xticks(deltas)
		fig.axes[0].set_xticklabels(self.names)
		fig.axes[0].set_ylabel('Distance Modulus (mag)',fontsize=self.FS)
		fig.axes[0].tick_params(labelsize=self.FS)
		fig.axes[0].set_ylim([Ylow-0.02,None])
		fig.axes[0].set_xlim([deltas[0]-mini_d/2,deltas[-1]+mini_d/2])
		fig.axes[0].set_xlabel('Siblings',fontsize=self.FS+3)
		pl.tight_layout()
		if self.save:
			pl.savefig(f"{self.plotpath}{self.galname}_IndividualDistances.pdf", bbox_inches="tight")
		if self.show:
			pl.show()

	def plot_sigmaRel_posteriors(self,xupperlim=0.25,colours=None):
		"""
		Plot Sigma_Rel Posteriors

		Method to overlay each sigmaRel posterior from each choice of prior upper bound

		Parameters
		----------
		xupperlim : float
			define maximum x-value (i.e. sigmaRel value) on plot for visualisation purposes only

		colours : lst (optional; default=None)
			colours for each sigmaRel overlay, defaults to ['green','purple','goldenrod']

		Returns
		----------
		plot of sigmaRel posterior overlays
		"""


		alph = 0.2 ; dfs = 3
		if 'posteriors' not in list(self.__dict__.keys()):
			self.get_sigmaRel_posteriors()
		if colours is None: colours = ['green','purple','goldenrod']

		fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
		pl.title(r'$\sigma_{\rm{Rel}}$-Posterior from Individual Distance Estimates',fontsize=self.FS+1, weight='bold')
		for ip,prior_upper_bound in enumerate(self.prior_upper_bounds):
			ccc = colours[ip]
			self.posterior = self.posteriors[prior_upper_bound]
			self.sigRs     = self.sigRs_store[prior_upper_bound]
			#Plots
			fig.axes[0].plot(self.sigRs, self.posterior,c=ccc)
			fig.axes[0].plot([self.sigRs[-1],self.sigRs[-1]],[0,self.posterior[-1]],linestyle='--',c=ccc)
			#Quantiles
			sigma_68,index68 = self.get_quantile(0.68)
			sigma_95,index95 = self.get_quantile(0.95)
			fig.axes[0].fill_between(self.sigRs[0:index68+1],np.zeros(index68+1),self.posterior[:index68+1],color=ccc,alpha=alph*0.5)
			fig.axes[0].plot([sigma_68,sigma_68],[0,self.posterior[index68]],c=ccc)
			fig.axes[0].annotate(str(int(68))+str("%"),xy=(sigma_68,self.posterior[index68]+0.04*(self.posterior[0]-self.posterior[-1])), color=ccc,fontsize=self.FS-4,weight='bold',ha='left')
			fig.axes[0].fill_between(self.sigRs[0:index95+1],np.zeros(index95+1),self.posterior[:index95+1],color=ccc,alpha=alph*(1-0.5)*0.5)
			fig.axes[0].plot([sigma_95,sigma_95],[0,self.posterior[index95]],c=ccc)
			fig.axes[0].annotate(str(int(95))+str("%"),xy=(sigma_95,self.posterior[index95]+0.04*(self.posterior[0]-self.posterior[-1])), color=ccc,fontsize=self.FS-4,weight='bold',ha='left')
			#Annotations
			Xoff = 0.75 ; dX   = 0.08 ; Yoff = 0.845 ; dY   = 0.07
			fig.axes[0].annotate('Prior Distribution',xy=(Xoff-0.06*1.55,Yoff+dY),xycoords='axes fraction',fontsize=15.5)
			fig.axes[0].annotate('Posterior Summary',xy=(Xoff-0.06*1.55,Yoff+dY-0.275001*1.125),xycoords='axes fraction',fontsize=15.5)
			LABEL = r'$\sigma_{\rm{Rel}} \sim U (0,%s)$'%(str(round(float(prior_upper_bound),3)))
			fig.axes[0].annotate(LABEL,xy=(Xoff-0.06*1.55,Yoff-dY*ip),xycoords='axes fraction',fontsize=self.FS-dfs,color=ccc)
			Yoff += -0.275001*1.125
			fig.axes[0].annotate("%s <"%r'$\sigma_{\rm{Rel}}$',       xy=(Xoff-0.01    ,Yoff-dY*ip),xycoords='axes fraction',fontsize=self.FS-dfs,color=ccc,ha='right')
			fig.axes[0].annotate("{:.3f}".format(sigma_68),  xy=(Xoff+dX+0.02,Yoff-dY*ip),xycoords='axes fraction',fontsize=self.FS-dfs,color=ccc,ha='right')
			fig.axes[0].annotate("({:.3f})".format(sigma_95),xy=(Xoff+dX+0.0313,Yoff-dY*ip),xycoords='axes fraction',fontsize=self.FS-dfs,color=ccc,ha='left')
			Yoff +=  0.275001
		fig.axes[0].set_yticks([])
		YMIN,YMAX = list(pl.gca().get_ylim())[:]
		pl.gca().set_xlim([0,xupperlim])
		fig.axes[0].set_ylim([0,YMAX])
		fig.text(0, 0.5, 'X', rotation=90, va='center', ha='center',color='white',fontsize=100)
		fig.text(-0.06, 0.5, 'Posterior Density', rotation=90, va='center', ha='center',color='black',fontsize=self.FS)
		fig.axes[0].set_xlabel(r'$\sigma_{\rm{Rel}}$ (mag)',fontsize=self.FS)
		pl.tick_params(labelsize=self.FS)
		pl.tight_layout()
		fig.axes[0].set_xticks(np.arange(0,0.25,0.05))
		if self.save:
			pl.savefig(f"{self.plotpath}{self.galname}_SigmaRelPosteriors.pdf",bbox_inches="tight")
		if self.show:
			pl.show()

	def plot_common_distances(self,markersize=10,capsize=8,mini_d=0.025):
		"""
		Plot Common Distances

		Parameters
		----------
		markersize : float (optional; default=10)
			size of distance scatter point markers

		capsize : float (optional; default=8)
			size of errorbar caps

		mini_d : float (optionl; default=0.025)
			delta_x value for separating distances visually
		"""

		fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
		fig.axes[0].set_title(r"Combination of Individual Distance Estimates",weight='bold',fontsize=self.FS+0.5)

		Nmodes = int(len(self.common_distance_estimates))
		deltas = [mini_d*(nn-(Nmodes-1)/2) for nn in range(Nmodes)]
		markers = [m for m in ['o','^','s'][:Nmodes]]
		colors  = ['r','g','b']
		mode_labels = [f"$\delta M$-{mode.capitalize()}" for mode in self.common_distance_estimates]
		sig_labels  = [r'$\sigma_{\rm{Rel}}=\sigma_0$',
					   r'$\sigma_{\rm{Rel}}\sim U(0,\sigma_0)$',
					   r'$\sigma_{\rm{Rel}}=0$']

		mus  = np.array([self.common_distance_estimates[mode]['median'] for mode in self.common_distance_estimates])
		errs = np.array([self.common_distance_estimates[mode]['std']    for mode in self.common_distance_estimates])
		alph=0.2
		fig.axes[0].plot(deltas,mus+errs,marker='None',c='black',alpha=alph)
		fig.axes[0].plot(deltas,mus     ,marker='None',c='black',alpha=alph)
		fig.axes[0].plot(deltas,mus-errs,marker='None',c='black',alpha=alph)
		fig.axes[0].plot(deltas,(mus[0]+errs[0])*np.ones(Nmodes),marker='None',c='r',alpha=alph,linestyle='--')
		fig.axes[0].plot(deltas,(mus[0]        )*np.ones(Nmodes),marker='None',c='r',alpha=alph,linestyle='--')
		fig.axes[0].plot(deltas,(mus[0]-errs[0])*np.ones(Nmodes),marker='None',c='r',alpha=alph,linestyle='--')

		for mode,nn,mu,err in zip(self.common_distance_estimates,np.arange(Nmodes),mus,errs):
			fig.axes[0].errorbar(deltas[nn],mu,yerr=err, color=colors[nn],marker=markers[nn],markersize=markersize,linestyle='none',capsize=capsize,label=mode_labels[nn],elinewidth=2)
			fig.axes[0].set_xticklabels("")
			if mode=='common':
				ylow    = mu-err*1.15
				yhigh   = mu+err*1.1
				dy      = 0.02

		for nn,mu,err in zip(np.arange(Nmodes),mus,errs):
			fig.axes[0].annotate(f'{mu:.3f}'+r'$\pm$'+f'{err:.3f}',#+'(mag)',
								 xy=(deltas[nn]-mini_d*0.4475,ylow-dy*3),color='black',fontsize=self.FS)
			fig.axes[0].annotate(sig_labels[nn],
								 xy=(deltas[nn],yhigh),color='black',fontsize=self.FS,ha='center')
		fig.axes[0].annotate(r'$\sigma_{\rm{Rel}}$ Priors:',
							xy=(deltas[1]-mini_d*0,yhigh+dy*1.5),color='black',fontsize=self.FS+1,weight='bold',ha='center')
		fig.axes[0].annotate(r'Common-$\mu$ Posteriors (mag):',
							xy=(deltas[1]-mini_d*0,ylow-dy*1.5),color='black',fontsize=self.FS+1,weight='bold',ha='center')

		fig.axes[0].set_xticks(deltas)
		fig.axes[0].set_xticklabels(mode_labels)
		fig.axes[0].set_ylabel('Distance Modulus (mag)',fontsize=self.FS)
		fig.axes[0].tick_params(labelsize=self.FS)
		fig.axes[0].set_ylim([ylow-dy*4,yhigh+dy*3])
		fig.axes[0].set_xlim([deltas[0]-mini_d/2,deltas[-1]+mini_d/2])
		fig.axes[0].set_xlabel('Intrinsic Scatter Modelling Assumption',fontsize=self.FS+3)
		pl.tight_layout()
		if self.save:
			pl.savefig(f"{self.plotpath}{self.galname}_CommonDistances.pdf", bbox_inches="tight")
		if self.show:
			pl.show()
