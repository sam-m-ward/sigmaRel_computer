"""
Sigma_Rel Model

Module containing multi_galaxy class and siblings_galaxy class
multi_galaxy class takes individual siblings distance estimates and performs multi-galaxy analysis
siblings_galaxy class same as multi_galaxy class but just for a single siblings galaxy

Contains:
--------------------
multi_galaxy class:
	inputs: dfmus,samplename='multigal',sigma0=0.1,sigmapec=250,eta_sigmaRel_input=None,use_external_distances=False,rootpath='./'

	Methods are:
		create_paths()
		update_attributes(other_class,attributes_to_add = ['modelkey','sigma0','sigmapec','sigmaRel_input','eta_sigmaRel_input','use_external_distances'])
		get_parlabels(pars)
		sigmaRel_sampler(sigma0=None, sigmapec=None, eta_sigmaRel_input=None, use_external_distances=None, overwrite=True)
		plot_posterior_samples(FS=18,paperstyle=True,quick=True,save=True,show=False, returner=False):
		compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],show=False,save=True)
		loop_single_galaxy_analyses()

siblings_galaxy class:
	inputs: mus,errors,names,galname,prior_upper_bounds=[0.1,0.15,1.0],sigma0=0.094,Ngrid=1000,fontsize=18,show=False,save=True

	Methods are:
		create_paths()
		get_sigmaRel_posterior(prior_distribution='uniform', prior_upper_bound = 1.0)
		get_sigmaRel_posteriors()
		get_CDF()
		get_quantile(q, return_index=True)
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
from model_loader_script import *
from plotting_script import *

class multi_galaxy_siblings:

	def create_paths(self):
		"""
		Create Paths

		Creates new folders if they don't already exist

		End Product(s)
		----------
		os.mkdir(self.path)
		"""
		for path in [self.productpath,self.plotpath]:
			minipaths = path.split('/')
			for _ in range(len(minipaths)-2):
				newpath = '/'.join(minipaths[:2+_]) + '/'
				if not os.path.exists(newpath):
					os.mkdir(newpath)

	def __init__(self,dfmus,samplename='multigal',sigma0=0.1,sigmapec=250,eta_sigmaRel_input=None,use_external_distances=False,rootpath='./'):
		"""
		Initialisation

		Parameters
		----------
		dfmus : pandas df
			dataframe of individual siblings distance estimates with columns Galaxy, SN, mus, mu_errs

		samplename : str (optional; default='multigal')
			name of multi-galaxy sample of siblings

		sigma0 : float (optional; default=0.1)
			the total intrinsic scatter, i.e. informative prior upper bound on sigmaRel

		sigmapec : float, str or None (optional; default=250)
			same as for sigma0, default value is 250 km/s

		eta_sigmaRel_input : float, or None (optional; default=None)
			option to fix sigmaRel at a fraction of sigma0
			if None, free sigmaRel
			if float, assert between 0 and 1

		use_external_distances : bool (optional; default=False)
			option to include external distance constraints

		rootpath : str
			path/to/sigmaRel/rootpath
		"""
		#Data
		self.dfmus      = dfmus
		self.samplename = samplename

		#Model
		self.sigma0                 = sigma0
		self.sigmapec               = sigmapec
		self.eta_sigmaRel_input     = eta_sigmaRel_input
		self.use_external_distances = use_external_distances

		#Paths
		self.rootpath    = rootpath
		self.modelpath   = self.rootpath  + 'model_files/'
		self.stanpath    = self.modelpath + 'stan_files/MultiGalFiles/'
		self.productpath = self.rootpath  + 'products/multigal/'
		self.plotpath    = self.rootpath  + 'plots/multi_galaxy_plots/'
		self.create_paths()

		#Posterior Configuration
		self.n_warmup   = 1000
		self.n_sampling = 5000
		self.n_chains   = 4
		self.n_thin     = 1000

		#Other
		self.c_light = 299792458

	def update_attributes(self,other_class,attributes_to_add = ['modelkey','sigma0','sigmapec','sigmaRel_input','eta_sigmaRel_input','use_external_distances']):
		"""
		Updated Attributes

		Method to update modelloader attributes to self.attributes

		Parameters
		----------
		other_class : class
			class with attributes to copy over

		attributes_to_add : lst
			names of attributes to copy over

		End Product(s)
		----------
		self.__dict__ updated with other_class.__dict__ attributes in attributes_to_add
		"""
		for x in attributes_to_add:
			self.__dict__[x] = other_class.__dict__[x]

	def get_parlabels(self,pars):
		"""
		Get Parameter Labels

		Method to grab names of parameters, their labels as they appear in posterior samples and in plots, and their prior lower/upper bounds

		Parameters
		----------
		pars : lst
		 	names of parameters to keep

		End Product(s)
		----------
		self.parnames, self.dfparnames, self.parlabels, self.bounds
		"""
		#Initialisation
		parnames   = ['sigmaRel','sigma0','sigmapec']
		dfparnames = ['sigmaRel','sigma0','sigmapec']
		parlabels  = ['$\\sigma_{\\rm{Rel}}$ (mag)','$\\sigma_0$ (mag)','$\\sigma_{\\rm{pec}}$ (km$\,$s$^{-1}$)']
		bounds     = [[0,None],[0,1],[0,self.c_light]]

		#Filter on pars
		PARS  = pd.DataFrame(data=dict(zip(['dfparnames','parlabels','bounds'],[dfparnames,parlabels,bounds])),index=parnames)
		PARS = PARS.loc[pars]

		#Set class attributes
		self.parnames   = pars
		self.dfparnames = PARS['dfparnames'].values
		self.parlabels  = PARS['parlabels'].values
		self.bounds     = PARS['bounds'].values

		if len(pars)==1 and pars[0]=='sigmaRel':
			self.bounds = [0,self.sigma0]
			print (f'Updating sigmaRel bounds to {self.bounds} seeing as sigma0/sigmapec are fixed')

	def sigmaRel_sampler(self, sigma0=None, sigmapec=None, eta_sigmaRel_input=None, use_external_distances=None, overwrite=True):
		"""
		sigmaRel Sampler

		Core method, used to take individual siblings distance estimates to multiple galaxies, and compute posteriors on sigmaRel
		Options to:
			- fix or free sigma0å
			- fix or free correlation coefficient rho = sigmalRel/sigma0
			- fix or free sigmapec
			- include external distance constraints

		Parameters
		----------
		sigma0 : float, str or None (optional; default=None)
			if None, set to self.sigma0
			if float, fix to float and assert>0
			if str, must be 'free'

		sigmapec : float, str or None (optional; default=None)
			same as for sigma0, default value is 250

		eta_sigmaRel_input : float, or None (optional; default=None)
			option to fix sigmaRel at a fraction of sigma0
			if None, free sigmaRel
			if float, assert between 0 and 1

		use_external_distances : bool (optional; default=False)
			option to include external distance constraints

		overwrite : bool (optional; default=True)
			option to overwrite previously saved .pkl posterior samples

		End Product(s)
		----------
		self.FIT, posterior samples saved in self.productpath
		"""
		#Initialise model with choices
		modelloader = ModelLoader(sigma0, sigmapec, eta_sigmaRel_input, use_external_distances, self)
		#Get values, either default or user input
		modelloader.get_model_params()
		#Get string name for creating/saving files and models
		modelloader.get_modelkey()
		#Get specific stan model file
		modelloader.update_stan_file()
		#Update class with current values
		self.update_attributes(modelloader)

		#If files don't exist or overwrite, do stan fits
		if not os.path.exists(self.productpath+f"FIT{self.samplename}{self.modelkey}.pkl") or overwrite:
			#Data
			dfmus = copy.deepcopy(self.dfmus)
			#Get stan_data
			stan_data   = modelloader.get_stan_data(dfmus)
			self.pars   = modelloader.get_pars()

			#Initialise and fit model
			model       = CmdStanModel(stan_file=self.stanpath+'current_model.stan')
			fit         = model.sample(data=stan_data,chains=self.n_chains, iter_sampling=self.n_sampling, iter_warmup = self.n_warmup)
			df          = fit.draws_pd()

			#Thin samples
			if df.shape[0]>self.n_thin*self.n_chains:#Thin samples
				print (f'Thinning samples down to {self.n_thin} per chain to save on space complexity')
				Nthinsize = int(self.n_thin*self.n_chains)
				df = df.iloc[0:df.shape[0]:int(df.shape[0]/Nthinsize)]						#thin to e.g. 1000 samples per chain
				dfdict = df.to_dict(orient='list')											#change to dictionary so key,value is parameter name and samples
				fit = {key:np.array_split(value,self.n_chains) for key,value in dfdict.items()}	#change samples so split into chains

			#Fitsummary including Rhat valuess
			fitsummary = az.summary(fit)#feed dictionary into arviz to get summary stats of thinned samples

			#Save samples
			FIT = dict(zip(['data','summary','chains','modelloader'],[stan_data,fitsummary,df]))
			with open(self.productpath+f"FIT{self.samplename}{self.modelkey}.pkl",'wb') as f:
				pickle.dump(FIT,f)

			#Print posterior summaries
			print (f"Fit Completed; Summary is:")
			for col in self.pars:
				print ('###'*10)
				print (f'{col}, median, std, 16, 84 // 68%, 95%')
				print (df[col].median().round(3),df[col].std().round(3),df[col].quantile(0.16).round(3),df[col].quantile(0.84).round(3))
				print (df[col].quantile(0.68).round(3),df[col].quantile(0.95).round(3))
				print ('Rhat:', fitsummary.loc[col]['r_hat'])

		else:#Else load up
			with open(self.productpath+f"FIT{self.samplename}{self.modelkey}.pkl",'rb') as f:
				FIT = pickle.load(f)

		#Print distance summaries
		self.FIT  = FIT

	def plot_posterior_samples(self, FS=18,paperstyle=True,quick=True,save=True,show=False, returner=False):
		"""
		Plot Posterior Samples

		Method to take posterior samples and plot them up

		Parameters
		----------
		FS : float (optional; default=18)
			fontsize for plotting

		paperstyle : bool (optional; default=True)
			if False, plot up Rhat values

		quick : bool (optional; default=True)
			if True, plot samples in 2D panels, if False, plot KDEs in 2D panel

		show : bool (optional; default=False)
			bool to show plots or not

		save : bool (optional; default=True)
			bool to save plots or not

		returner : bool (optional; default=False)
			if True, return posterior summaries for use in a Table

		End Product(s)
		----------
		Plot
		"""

		#Get pars to plot
		if 'modelkey' not in self.__dict__:
			modelloader = ModelLoader(self.sigma0, self.sigmapec, self.eta_sigmaRel_input, self.use_external_distances, self)
			modelloader.get_model_params()
			modelloader.get_modelkey()
			self.update_attributes(modelloader)
			self.pars   = modelloader.get_pars()
		self.get_parlabels(self.pars)

		#Get posterior samples
		if 'FIT' in self.__dict__.keys():
			FIT = self.FIT
		else:
			savekey  = self.samplename+self.modelkey
			filename = self.productpath+f"FIT{savekey}.pkl"
			with open(filename,'rb') as f:
				FIT = pickle.load(f)

		#Get posterior data
		df         = FIT['chains'] ; fitsummary = FIT['summary'] ; stan_data  = FIT['data']
		Rhats      = {par:fitsummary.loc[par]['r_hat'] for par in self.dfparnames}
		samples    = {par:np.asarray(df[par].values)   for par in self.dfparnames}
		print ('Rhats:',Rhats)

		#Corner Plot
		self.plotting_parameters = {'FS':FS,'paperstyle':paperstyle,'quick':quick,'save':save,'show':show}
		postplot = POSTERIOR_PLOTTER(samples, self.parnames, self.parlabels, self.bounds, Rhats, self.plotting_parameters)
		Summary_Strs = postplot.corner_plot()#Table Summary
		savekey         = self.samplename+self.modelkey+'_FullKDE'*bool(not self.plotting_parameters['quick'])+'_NotPaperstyle'*bool(not self.plotting_parameters['paperstyle'])
		save,quick,show = [self.plotting_parameters[x] for x in ['save','quick','show']][:]
		finish_corner_plot(postplot.fig,postplot.ax,get_Lines(stan_data,self.c_light),save,show,self.plotpath,savekey)

		#Return posterior summaries
		if returner: return Summary_Strs



	def compute_analytic_multi_gal_sigmaRel_posterior(self,prior_upper_bounds=[1.0],show=False,save=True):
		"""
		Compute Analytic Multi Galaxy sigmaRel Posterior

		Method to compute analytic sigmaRel posterior by multiplying the single-galaxy likelihoods by the prior

		Parameters
		----------
		prior_upper_bounds : list (optional; default=[1.0])
			choices of sigmaRel prior upper bound

		show : bool (optional; default=False)
			bool to show plots or not

		save : bool (optional; default=True)
			bool to save plots or not

		End Product(s)
		----------
		Plot of multi-galaxy sigmaRel posterior

		self.total_posteriors: dict
			key,value pairs are the sigmaRel prior upper bound, and the posterior

		self.sigRs_store: dict
		 	same as self.total_posteriors, but the sigR prior grid
		"""
		#List of prior upper bounds to loop over
		self.prior_upper_bounds = prior_upper_bounds

		#Initialise posteriors
		total_posteriors = {p:1 for p in self.prior_upper_bounds} ; sigRs_store = {}
		#For each galaxy, compute sigmaRel likelihood
		for g,gal in enumerate(self.dfmus['Galaxy'].unique()):
			dfgal  = self.dfmus[self.dfmus['Galaxy']==gal]												#Use dummy value for sigma0
			sibgal = siblings_galaxy(dfgal['mus'].values,dfgal['mu_errs'].values,dfgal['SN'].values,gal,sigma0=0.1,prior_upper_bounds=self.prior_upper_bounds)
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
		sibgal.show     = show
		sibgal.save     = save
		sibgal.galname  = self.samplename
		sibgal.plotpath = self.plotpath
		sibgal.plot_sigmaRel_posteriors(xupperlim='adaptive')

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

	def create_paths(self):
		"""
		Create Paths

		Creates new folders if they don't already exist

		End Product(s)
		----------
		os.mkdir(self.path)
		"""
		for path in [self.productpath,self.plotpath]:
			minipaths = path.split('/')
			for _ in range(len(minipaths)-2):
				newpath = '/'.join(minipaths[:2+_]) + '/'
				if not os.path.exists(newpath):
					os.mkdir(newpath)

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
		self.create_paths()

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

	def get_quantile(self, q, return_index=True):
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

		return_index : bool (optional; default=True)
			if True, return the index of the quantile
		"""
		self.get_CDF()
		index_q = np.argmin(np.abs(self.CDF-q))
		sigR_q  = self.sigRs[index_q]
		if return_index:
			return sigR_q, index_q
		else:
			return sigR_q

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
		xupperlim : float or 'adaptive' (optional; default=0.25)
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
		XQs = {0.005:[],0.995:[]}
		for ip,prior_upper_bound in enumerate(self.prior_upper_bounds):
			ccc = colours[ip]
			self.posterior = self.posteriors[prior_upper_bound]
			self.sigRs     = self.sigRs_store[prior_upper_bound]
			#Plots
			fig.axes[0].plot(self.sigRs, self.posterior,c=ccc)
			fig.axes[0].plot([self.sigRs[-1],self.sigRs[-1]],[0,self.posterior[-1]],linestyle='--',c=ccc)
			#Begin Annotations
			Xoff = 0.65 ; dX   = 0.08 ; Yoff = 0.845 ; dY   = 0.07
			fig.axes[0].annotate('Prior Distribution',xy=(Xoff,Yoff+dY),			   xycoords='axes fraction',fontsize=15.5, ha='left')
			fig.axes[0].annotate('Posterior Summary', xy=(Xoff,Yoff+dY-0.275001*1.125),xycoords='axes fraction',fontsize=15.5, ha='left')
			LABEL = r'$\sigma_{\rm{Rel}} \sim U (0,%s)$'%(str(round(float(prior_upper_bound),3)))
			fig.axes[0].annotate(LABEL,xy=(Xoff,Yoff-dY*ip),xycoords='axes fraction',fontsize=self.FS-dfs,color=ccc,ha='left')
			#Decide how to summarise posterior
			KDE = copy.deepcopy(self.posterior)
			imode = np.argmax(KDE)
			xmode = self.sigRs[imode] ; KDEmode = KDE[imode]
			condition1 = np.argmax(KDE)!=0 and np.argmax(KDE)!=len(KDE)-1#KDE doesnt peak at prior boundary
			hh = np.exp(-1/8)#Gaussian height at sigma/2 #KDE is not near flat topped at prior boundary
			condition2 = not (KDE[0]>=hh*KDEmode or KDE[-1]>=hh*KDEmode)
			Yoff += -0.275001*1.125
			for q in XQs:
				XQs[q].append(self.get_quantile(q)[0])
			if condition1 and condition2:
				sigma_50,index50 = self.get_quantile(0.50)
				fig.axes[0].plot(np.ones(2)*self.sigRs[index50],[0,KDE[index50]],c=ccc)
				sigma_16,index16 = self.get_quantile(0.16)
				sigma_84,index84 = self.get_quantile(0.84)
				index84 += 1
				fig.axes[0].fill_between(self.sigRs[index16:index84],np.zeros(index84-index16),KDE[index16:index84],color=ccc,alpha=alph)
				summary = ["{:.3f}".format(x) for x in [self.sigRs[index50],self.sigRs[index84-1]-self.sigRs[index50],self.sigRs[index50]-self.sigRs[index16]] ]
				fig.axes[0].annotate(r"$\sigma_{\rm{Rel}} = %s ^{+%s}_{-%s}$"%(summary[0],summary[1],summary[2]),xy=(Xoff,Yoff-dY*ip),xycoords='axes fraction',fontsize=self.FS-dfs,color=ccc,ha='left')
			else:
				#Quantiles
				if imode>0.5*(len(self.sigRs)-1):#If peaks at RHS
					sigma_68,index68 = self.get_quantile(1-0.68)
					sigma_95,index95 = self.get_quantile(1-0.95)
					fig.axes[0].fill_between(self.sigRs[index68:],np.zeros(len(self.sigRs)-index68),self.posterior[index68:],color=ccc,alpha=alph*0.5)
					fig.axes[0].fill_between(self.sigRs[index95:],np.zeros(len(self.sigRs)-index95),self.posterior[index95:],color=ccc,alpha=alph*(1-0.5)*0.5)
					lg = '>'
				else:
					sigma_68,index68 = self.get_quantile(0.68)
					sigma_95,index95 = self.get_quantile(0.95)
					fig.axes[0].fill_between(self.sigRs[0:index68+1],np.zeros(index68+1),self.posterior[:index68+1],color=ccc,alpha=alph*0.5)
					fig.axes[0].fill_between(self.sigRs[0:index95+1],np.zeros(index95+1),self.posterior[:index95+1],color=ccc,alpha=alph*(1-0.5)*0.5)
					lg = '<'

				fig.axes[0].plot([sigma_68,sigma_68],[0,self.posterior[index68]],c=ccc)
				fig.axes[0].plot([sigma_95,sigma_95],[0,self.posterior[index95]],c=ccc)
				fig.axes[0].annotate(str(int(68))+str("%"),xy=(sigma_68,self.posterior[index68]+0.04*(self.posterior[0]-self.posterior[-1])), color=ccc,fontsize=self.FS-4,weight='bold',ha='left')
				fig.axes[0].annotate(str(int(95))+str("%"),xy=(sigma_95,self.posterior[index95]+0.04*(self.posterior[0]-self.posterior[-1])), color=ccc,fontsize=self.FS-4,weight='bold',ha='left')
				#Continue Annotations
				s68 = "{:.3f}".format(sigma_68) ; s95 = "({:.3f})".format(sigma_95)
				fig.axes[0].annotate("%s %s %s %s"%(r'$\sigma_{\rm{Rel}}$',lg, s68, s95),xy=(Xoff,Yoff-dY*ip),xycoords='axes fraction',fontsize=self.FS-dfs,color=ccc,ha='left')
			Yoff +=  0.275001

		fig.axes[0].set_yticks([])
		YMIN,YMAX = list(pl.gca().get_ylim())[:]
		if xupperlim!='adaptive':
			pl.gca().set_xlim([0,xupperlim])
			fig.axes[0].set_xticks(np.arange(0,0.25,0.05))
		else:
			qmin,qmax = min(list(XQs.keys())),max(list(XQs.keys()))
			DX = max(XQs[qmax]) - min(XQs[qmin]) ; fac = 0.1
			pl.gca().set_xlim([max([0,min(XQs[qmin])-DX*fac]),max(XQs[qmax])+DX*2*fac])
		fig.axes[0].set_ylim([0,YMAX])
		fig.axes[0].set_ylabel(r'Posterior Density',fontsize=self.FS)
		fig.axes[0].set_xlabel(r'$\sigma_{\rm{Rel}}$ (mag)',fontsize=self.FS)#fig.text(0, 0.5, 'X', rotation=90, va='center', ha='center',color='white',fontsize=100)#fig.text(-0.06, 0.5, 'Posterior Density', rotation=90, va='center', ha='center',color='black',fontsize=self.FS)
		pl.tick_params(labelsize=self.FS)
		pl.tight_layout()
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
