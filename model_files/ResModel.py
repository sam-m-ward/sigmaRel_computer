"""
Residual Model

Module containing ResModelLoader class
ResModelLoader class used to perform multiple linear regression between parameters

Contains:
--------------------
ResModelLoader class:
	inputs: dfmus,rootpath='./',FS=18,verbose=True

	Methods are:
		plot_res(p1='theta',p2='mu',show=False,save=True)
		get_data()
		get_inits()
		prepare_stan()
		get_parlabels()
		get_modelkey()
		sample_posterior(alpha=False,py='mu',pxs=['theta'], alpha_prior=10, beta_prior=10, sigint_prior=10, beta=None, overwrite=True)
		get_parlabels()
		plot_posterior_samples(FS=None,paperstyle=True,quick=True,save=True,show=False, returner=False)
--------------------

Written by Sam M. Ward: smw92@cam.ac.uk
"""
import arviz as az
from cmdstanpy import CmdStanModel
import json, os, pickle, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from plotting_script import *


class ResModelLoader:
	def __init__(self,dfmus,rootpath='./',FS=18):
		"""
		Initialisation

		Parameters
		----------
		dfmus : pandas df
			dataframe of parameters with columns Galaxy, SN, {PAR}_samps

		rootpath : str
			path/to/sigmaRel/rootpath

		FS : float (optional; default=18)
			fontsize for plots
		"""
		#Data
		self.dfmus = dfmus

		#Paths
		self.rootpath    = rootpath
		self.modelpath   = self.rootpath  + 'model_files/'
		self.stanpath    = self.modelpath + 'stan_files/linear_reg/'
		self.plotpath    = self.rootpath  + 'plots/linear_reg/'
		self.productpath = self.rootpath  + 'products/linear_reg/'

		self.FS = FS

		#Create res samples for plotting
		self.PARS = [par for par in ['mu','AV','theta','RV','etaAV'] if f'{par}_samps' in self.dfmus.columns]
		def mean_mapper(x):
			mean_samps = x.mean(axis=0)
			x = x.apply(lambda x: x-mean_samps)
			return x
		for PAR in self.PARS:
			self.dfmus[f'{PAR}res_samps'] = self.dfmus.groupby('Galaxy')[f'{PAR}_samps'].transform(lambda x: mean_mapper(x))
			self.dfmus[f'{PAR}_respoint'] = self.dfmus[f'{PAR}res_samps'].apply(np.median)
			self.dfmus[f'{PAR}_reserr']   = self.dfmus[f'{PAR}res_samps'].apply(np.std)
		self.dfmus = self.dfmus.iloc[::2].copy()
		print (self.dfmus[[col for col in dfmus.columns if 'respoint' in col or 'reserr' in col] + ['Galaxy']])

		#Input Posterior Parameters
		self.master_parnames  = ['mu','AV','theta','RV','etaAV']
		self.master_parlabels = ['$\\Delta \\mu$ (mag)','$\\Delta A_V$ (mag)','$\\Delta \\theta$','$\\Delta \\R_V$','$\\Delta \\eta_{A_V}$']
		self.master_bounds    = [[None,None],[0,None],[None,None],[None,None],[None,None]]

		#Posterior Configuration
		self.n_warmup   = 10000
		self.n_sampling = 10000
		self.n_chains   = 4
		self.n_thin     = 1000

	def plot_res(self,p1='theta',p2='mu',show=False,save=True):
		"""
		Plot Residuals

		Plots the residuals of two parameters against one another

		Parameters
		----------
		p1,p2 : strs (optional; default='theta', 'mu')
			the x and y parameters, respectively

		show, save: bools (optional; default=False,True)
			whether to show/save plot

		End Product
		----------
		Plot of residuals
		"""
		dfmus = self.dfmus.copy()
		colours = [f'C{s%10}' for s in range(dfmus['Galaxy'].nunique())]
		markers = ['o','s','p','^','x','P','d','*']
		markers = [markers[int(ig%len(markers))] for ig in range(dfmus['Galaxy'].nunique())]
		pl.figure()
		for ig,g in enumerate(dfmus['Galaxy'].unique()):
			z1m = dfmus[dfmus['Galaxy']==g][f'{p1}res_samps'].apply(np.median)
			z2m = dfmus[dfmus['Galaxy']==g][f'{p2}res_samps'].apply(np.median)
			z1e = dfmus[dfmus['Galaxy']==g][f'{p1}res_samps'].apply(np.std)
			z2e = dfmus[dfmus['Galaxy']==g][f'{p2}res_samps'].apply(np.std)
			pl.errorbar(z1m,z2m,z1e,z2e,linestyle='none',marker=markers[ig],color=colours[ig])
		pl.xlabel(self.master_parlabels[self.master_parnames.index(p1)], fontsize=self.FS)
		pl.ylabel(self.master_parlabels[self.master_parnames.index(p2)], fontsize=self.FS)
		pl.tight_layout()
		pl.tick_params(labelsize=self.FS)
		if save:
			pl.savefig(self.plotpath+f'resplot_{p1}_{p2}.pdf',bbox_inches="tight")
		if show:
			pl.show()

	def get_data(self):
		"""
		Get Data

		Method for getting stan data

		End Products
		----------
		stan_data dictionary
		"""
		Nd   = self.dfmus.shape[0]
		Nf   = len(self.pxs)
		Y    = self.dfmus[f'{self.py}_respoint'].to_numpy()
		Yerr = self.dfmus[f'{self.py}_reserr'].to_numpy()
		X    = self.dfmus[[f'{px}_respoint' for px in self.pxs]].to_numpy()
		Xerr = self.dfmus[[f'{px}_reserr' for px in self.pxs]].to_numpy()


		'''#Simulated data for testing stan model
		Nd     = 100
		betas  = np.array([1 for _ in range(Nf)])
		alpha  = 0
		sigint = 0.01
		x      = np.asarray([np.random.normal(0,5,Nd) for _ in range(Nf)]).T
		y      = x@betas + alpha + np.random.normal(0,sigint,Nd)
		X      = x+np.asarray([np.random.normal(0,0.01,Nd) for _ in range(Nf)]).T
		Y      = y+np.random.normal(0,0.01,Nd)
		Xerr   = np.asarray([np.ones(Nd)*0.01 for _ in range(Nf)]).T
		Yerr   = np.ones(Nd)*0.01
		#'''

		stan_data = dict(zip(
			['Nd','Nf','Y','Yerr','X','Xerr','beta_prior','sigint_prior'],
			[ Nd , Nf , Y , Yerr , X , Xerr , self.beta_prior , self.sigint_prior ]
		))
		if self.alpha:#Typically set intercept to zero for residuals
			stan_data['alpha_prior'] = self.alpha_prior
		else:
			stan_data['alpha_prior'] = 1e-6

		if self.beta is not None:#Sometimes freeze betas so input these as data
			stan_data['beta'] = self.beta

		return stan_data

	def get_inits(self):
		"""
		Get inits

		Simple initialisations for stan model

		End Product
		----------
		list of json files with initialisations that cmdstan can read
		"""
		json_file = {**{'alpha':0,'sigint':1e-6}, **{f'beta[{_+1}]':0 for _ in range(len(self.pxs))}}
		with open(f"{self.productpath}inits.json", "w") as f:
			json.dump(json_file, f)

		stan_init  = [f"{self.productpath}inits.json" for _ in range(self.n_chains)]
		return stan_init

	def prepare_stan(self):
		"""
		Prepare stan

		Get the .stan file

		End Products
		----------
		Prints out model and writes to current_model.stan
		"""
		#Update stan file according to choices, create temporary 'current_model.stan'
		self.stan_filename = 'linear_regression.stan'#{True:'linear_regression_QR.stan',False:'linear_regression.stan'}[self.QR]
		if self.beta is not None:
			self.stan_filename = 'linear_regression_fixedB.stan'

		with open(self.stanpath+self.stan_filename,'r') as f:
			stan_file = f.read().splitlines()
		stan_file = '\n'.join(stan_file)
		print (stan_file)
		with open(self.stanpath+'current_model.stan','w') as f:
			f.write(stan_file)

	def get_pars(self):
		"""
		Get Pars

		Gets the parameters used by model

		End Product
		----------
		pars : list of str
			the parameter names used/saved in model
		"""
		pars  = [f'beta[{_+1}]' for _ in range(len(self.pxs))]
		pars += ['sigint','alpha'] + ['sigmaRel']
		if self.alpha is False:
			pars.remove('alpha')
		if self.beta is not None:
			for _ in range(len(self.pxs)):
				pars.remove(f'beta[{_+1}]')
		return pars

	def get_modelkey(self):
		"""
		Get Model Key

		Takes input choices and builds a filename called modelkey

		End Product(s)
		----------
		self.modelkey : str
			descriptive filename of model choice
		"""

		print ('###'*10)
		self.modelkey = f"Y{self.py}__X{'_'.join(self.pxs)}_"
		if self.alpha:
			print ('Including intercept')
			self.modelkey += '_inclalpha'
		else:
			print ('Setting intercept to zero')
		if self.beta is None:
			print ('Fitting beta parameters')
			self.modelkey += '_fitbeta'
		else:
			print ('Fixing beta parameters to:', dict(zip(self.pxs,self.beta)))
			self.modelkey += f"_fixbeta{'_'.join([str(b) for b in self.beta])}"
		print ('###'*10)
		self.filename = f"FIT{self.modelkey}.pkl"

	def sample_posterior(self,alpha=False,py='mu',pxs=['theta'], alpha_prior=10, beta_prior=10, sigint_prior=10, beta=None, overwrite=True):
		"""
		Sample Posterior

		Method for running posterior sampling

		Parameters
		----------
		alpha : bool (optional; default=False)
			set whether intercept is zero or fitted, if False alpha=0

		py, pxs : str, list of str (optional; default='mu', ['theta'])
			predictor and features for multiple linear regression

		alpha_prior, beta_prior, sigint_prior : floats (optional; default=10,10,10)
			width of normal prior on hyperparameters, if alpha is None alpha_prior is set to 1e-6

		beta : None or list of floats (optional; default is None)
			if None, beta is fitted for, else, it is fixed at values given in list e.g. beta = [0.1]

		overwrite : bool (optional; default=True)
			if True, run new fit, else, load up previous fit

		End Product(s)
		----------
		runs stan fit, saves FIT in productpath
		assigns self.FIT
		"""
		self.alpha = alpha
		self.beta  = beta
		self.py    = py
		self.pxs   = pxs
		self.alpha_prior,self.beta_prior,self.sigint_prior = alpha_prior,beta_prior,sigint_prior
		self.get_modelkey()

		#If files don't exist or overwrite, do stan fits
		if not os.path.exists(self.productpath+self.filename) or overwrite:
			stan_data = self.get_data()
			stan_init = self.get_inits()
			self.prepare_stan()
			self.pars = self.get_pars()

			#Initialise and fit model
			model       = CmdStanModel(stan_file=self.stanpath+'current_model.stan')
			fit         = model.sample(data=stan_data,chains=self.n_chains, iter_sampling=self.n_sampling, iter_warmup = self.n_warmup, inits=stan_init, seed=42)
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
			FIT = dict(zip(['data','summary','chains'],[stan_data,fitsummary,df]))
			with open(self.productpath+self.filename,'wb') as f:
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
			with open(self.productpath+self.filename,'rb') as f:
				FIT = pickle.load(f)
			self.pars = self.get_pars()#For plotting

		#Print distance summaries
		self.FIT  = FIT

	def get_parlabels(self):
		"""
		Get Parlabels

		Simple Method for posterior plotting, gets the appropriate labels and filters down based on model fitting choices

		End Product(s)
		----------
		assigns: parnames, dfparnames, parlabels, bounds
		parnames are names in data, dfparnames are names in model, parlabels are those for plotting, bounds are hard prior bounds on parameters for reflection KDEs
		"""
		#Initialisation
		parnames   = [f'beta[{_+1}]' for _ in range(len(self.pxs))] + ['sigint','sigmaRel','alpha']
		dfparnames = [f'beta[{_+1}]' for _ in range(len(self.pxs))] + ['sigint','sigmaRel','alpha']
		pxdict     = {'theta':'\\theta','etaAV':'\\eta_{A_V}'}
		parlabels  = ['$\\beta_{%s}$'%(pxdict[px]) for px in self.pxs] + ['$\\sigma_{\\rm{int}}$','$\\sigma_{\\rm{Rel}}$','$\\alpha$']
		bounds     = [[None,None] for _ in range(len(self.pxs))] + [[0,None],[0,None],[None,None]]

		#Filter on pars
		PARS  = pd.DataFrame(data=dict(zip(['dfparnames','parlabels','bounds'],[dfparnames,parlabels,bounds])),index=parnames)
		PARS = PARS.loc[self.pars]

		#Set class attributes
		self.parnames   = self.pars
		self.dfparnames = PARS['dfparnames'].values
		self.parlabels  = PARS['parlabels'].values
		self.bounds     = PARS['bounds'].values

	def plot_posterior_samples(self, FS=None,paperstyle=True,quick=True,save=True,show=False, returner=False):
		"""
		Plot Posterior Samples

		Creates corner plot of fit

		Parameters
		----------
		FS : float or None (optional; default=None)
			fontsize for plotting, if None use self.FS

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
		if FS is None: FS = self.FS

		#Get pars to plot
		if 'modelkey' not in self.__dict__:
			raise Exception('Please run sample_posterior() with overwrite=False to initialise')
		self.get_parlabels()

		#Get posterior samples
		if 'FIT' in self.__dict__.keys():
			FIT = self.FIT
		else:
			with open(self.productpath+self.filename,'rb') as f:
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
		savekey         = self.modelkey+'_FullKDE'*bool(not self.plotting_parameters['quick'])+'_NotPaperstyle'*bool(not self.plotting_parameters['paperstyle'])
		save,quick,show = [self.plotting_parameters[x] for x in ['save','quick','show']][:]
		#finish_corner_plot(postplot.fig,postplot.ax,get_Lines(stan_data,self.c_light,modelloader.alt_prior),save,show,self.plotpath,savekey)
		finish_corner_plot(postplot.fig,postplot.ax,[],save,show,self.plotpath,savekey)

		#Return posterior summaries
		if returner: return Summary_Strs
