"""
Sigma_Rel Model

Module containing multi_galaxy class and siblings_galaxy class
multi_galaxy class takes individual siblings distance estimates and performs multi-galaxy analysis
siblings_galaxy class same as multi_galaxy class but just for a single siblings galaxy

Contains:
--------------------
multi_galaxy class:
	inputs: dfmus,samplename='full',sigma0=0.1,sigmapec=250,eta_sigmaRel_input=None,use_external_distances=False,zcosmo='zHD',rootpath='./',FS=18,verbose=True,fiducial_cosmology={'H0':73.24,'Om0':0.28}

	Methods are:
		create_paths()
		infer_pars()
		get_redshift_distances()
		get_cosmo_subsample(limits={'thetas':[-1.5,2.0],'AVs':[0.0,1.0]})
		print_table(PARS=['mu','AV','theta','RV'],verbose=False,returner=False)
		trim_sample(key='cosmo',bool_column=None)
		restore_sample()
		update_attributes(other_class,attributes_to_add = ['modelkey','sigma0','sigmapec','sigmaRel_input','eta_sigmaRel_input','use_external_distances'])
		get_parlabels(pars)
		sigmaRel_sampler(sigma0=None, sigmapec=None, eta_sigmaRel_input=None, use_external_distances=None, zmarg=False, alt_prior=False, overwrite=True, blind=False, zcosmo='zHD', alpha_zhel=False, Rhat_threshold=np.inf, Ntrials=1)
		add_transformed_params(df,fitsummary)
		plot_posterior_samples(   FS=18,paperstyle=True,quick=True,save=True,show=False, returner=False,blind=False,fig_ax=None,**kwargs)
		plot_posterior_samples_1D(FS=18,paperstyle=True,quick=True,save=True,show=False, returner=False,blind=False,fig_ax=None,**kwargs)
		compute_analytic_multi_gal_sigmaRel_posterior(PAR='mu',prior_upper_bounds=[1.0],alpha_zhel=False,show=False,save=True,blind=False,fig_ax=None,prior=None)
		loop_single_galaxy_analyses()
		get_dxgs(Sg,ss,g_or_z)
		plot_delta_HR(save=True,show=False)
		plot_parameters(PAR='mu',colours=None, markers=None, g_or_z = 'g',subtract_g_mean=None,zword='zHD_hats',show=False, save=True,markersize=14,capsize=5,alpha=0.9,elw=3,mew=3, plot_full_errors=False,plot_sigma0=0.094,plot_sigmapec=250,text_index = 3, annotate_mode = 'legend',args_legend={'loc':'upper left','ncols':2,'bbox_to_anchor':(1,1.02)}


siblings_galaxy class:
	inputs: mus,errors,names,galname,prior_upper_bounds=[0.1,0.15,1.0],sigma0=0.094,sigR_res=0.00025,fontsize=18,show=False,save=True

	Methods are:
		create_paths()
		get_sigmaRel_posterior(prior_distribution='uniform', prior_upper_bound = 1.0)
		get_sigmaRel_posteriors()
		get_CDF()
		get_quantile(q, return_index=True)
		get_weighted_mu()
		combine_individual_distances(mode=None,overwrite=True,asymmetric=False)
		plot_individual_distances(colours=None,markers=None,markersize=10,capsize=8,mini_d=0.025,plot_full_errors=True)
		plot_sigmaRel_posteriors(xupperlim=0.16,colours = ['green','purple','goldenrod'], blind=False, fig_ax=None)
		plot_common_distances(markersize=10,capsize=8,mini_d=0.025,asymmetric=None)
--------------------

Written by Sam M. Ward: smw92@cam.ac.uk
"""

import arviz as az
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as pl
import numpy as np
import copy, os, pickle, re, shutil, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_loader_script import *
from plotting_script import *
from astropy.cosmology import FlatLambdaCDM
from matplotlib import container

class multi_galaxy_siblings:

	def create_paths(self):
		"""
		Create Paths

		Creates new folders if they don't already exist

		End Product(s)
		----------
		os.mkdir(self.path)
		"""
		for path in [self.productpath,self.plotpath,self.modelpath]:
			minipaths = path.split('/')
			for _ in range(len(minipaths)-2):
				newpath = '/'.join(minipaths[:2+_]) + '/'
				if not os.path.exists(newpath):
					os.mkdir(newpath)

	def infer_pars(self):
		"""
		Simple Method to infer parameters from input dfmus

		End Product(s)
		----------
		self.parcols = ['mus','mu_errs','AVs','AV_errs'...]
		"""
		cols = list(self.dfmus.columns)
		pars = [cc.split('_errs')[0] for cc in cols if '_errs' in cc and f"{cc.split('_errs')[0]}s" in cols]
		self.parcols = [cc+{0:'s',1:'_errs'}[_] for cc in pars for _ in range(2)]

	def get_redshift_distances(self):
		"""
		Get Redshift Distances

		Take fiducial cosmology and compute mu_ext columns using redshifts

		End Product(s)
		----------
		self.cosmo, self.zcosmo_cols, self.zhelio_cols, self.mucosmo_cols:
			the astropy cosmology model,
			the redshift columns in dfmus of ['zcmb_hats','zHD_hats'] and ['zhelio_hats','zhelio_errs'], respectively
			the muext columns ['muext_zcmb_hats','muext_zHD_hats']] computed using cosmo
		"""
		self.cosmo = FlatLambdaCDM(**self.fiducial_cosmology)
		self.zcosmo_cols = [col for col in ['zcmb_hats','zHD_hats'] if col in self.dfmus.columns]
		self.zhelio_cols = [col for col in ['zhelio_hats','zhelio_errs'] if col in self.dfmus.columns]
		try:
			assert('zhelio_hats' in self.dfmus.columns)
		except:
			print ("Need 'zhelio_hats' column to compute redshift-based cosmology distances")
		for zcol in self.zcosmo_cols:
			assert(zcol.split('_hats')[0]+'_errs' in self.dfmus.columns)#Check errors in either/both of zHD and zcmb are there
			if f'muext_{zcol}' not in self.dfmus.columns:
				print (f"Computing muext_{zcol} from fiducial cosmology: {self.fiducial_cosmology}, and columns: 'zhelio_hats', '{zcol}'")
				self.dfmus[f'muext_{zcol}'] = self.dfmus[['zhelio_hats',zcol]].apply(lambda z: self.cosmo.distmod(z[1]).value + 5*np.log10((1+z[0])/(1+z[1])),axis=1)
			else:
				print (f'Using pre-supplied cosmology distances: muext_{zcol}')
		self.mucosmo_cols = [f'muext_{zcol}' for zcol in self.zcosmo_cols]

	def get_cosmo_subsample(self,limits={'thetas':[-1.5,2.0],'AVs':[0.0,1.0]}):
		"""
		Get Cosmology Subsample

		Take siblings sample and apply cuts on e.g. theta, AV

		Parameters
		----------
		limits : dict (optional; default={'theta':[-1.5,2.0],'AV':[0.0,1.0]})

		End Product(s)
		----------
		dfmus['cosmo_sample'] bool column
		"""
		missing_pars = [par for par in limits if par not in self.dfmus.columns]
		if len(missing_pars)==0:
			if self.verbose: print (f'Creating cosmo. sub-samp. using: {limits}')
			bools = {}
			for g in self.dfmus.Galaxy:
				dfg = self.dfmus[self.dfmus.Galaxy==g]
				bools[g] = True
				for par in limits:
					if par in dfg.columns and dfg[(dfg[par]<limits[par][0])|(dfg[par]>limits[par][1])].shape[0]>0:
						bools[g] = False
						break
			self.dfmus['cosmo_sample'] = self.dfmus['Galaxy'].map(bools)
		elif len(missing_pars)>0:
			if self.verbose:	print (f'Cosmo. sub-samp cannot be created by applying limits: {limits} because {missing_pars} missing from input columns;')
			if 'cosmo_sample' in self.dfmus:
				if self.verbose: print ("Instead use user-inputted 'cosmo_sample' bool column")
			else:
				if self.verbose: print ("Instead set all SNe as belonging to 'cosmo_sample'")
				self.dfmus.loc[:,'cosmo_sample'] = True

	def print_table(self, PARS=['mu','AV','theta','RV'],verbose=False,returner=False):
		"""
		Print Tables

		Method to take posterior chains and print summaries

		Parameters
		----------
		PARS: list (optional; default=['mu','AV','theta','RV'])
			total list that is then trimmed to match that in dfmus

		verbose : bool (optional; default=False)
			option to print out summaries for each plot

		returner: bool (optional; default=False)
			if True, return list of summaries

		End Product(s)
		----------
		Prints out string
		"""
		all_PARS   = copy.deepcopy(PARS)
		pPARS      = ['mu','AV','theta','RV']
		PARS       = [p for p in pPARS if p in PARS]
		pars       = [p for p in PARS if f'{p}s' in self.parcols]
		samp_cols  = [f'{p}_samps' for p in pars if f'{p}_samps' in self.dfmus.columns]

		keep        = [i for i,p in enumerate(self.master_parnames) if p in pars]
		parlabels   = [x for i,x in enumerate(self.master_parlabels) if i in keep]
		bounds      = [x for i,x in enumerate(self.master_bounds)    if i in keep]

		plotting_parameters = {'FS':self.FS,'paperstyle':False,'quick':True,'save':False,'show':False}
		lines = [] ; old_g = self.dfmus['Galaxy'].iloc[0]
		for isn,sn in enumerate(self.dfmus.index):
			NSN_g = str(self.dfmus[self.dfmus.Galaxy==self.dfmus.Galaxy.loc[sn]].shape[0])
			if self.dfmus['Galaxy'].loc[sn]!=old_g:
				lines.append('\midrule')
				old_g = self.dfmus['Galaxy'].loc[sn]
				new_g = True
			else:
				new_g = False if isn!=0 else True
			try:
				samples = {par:self.dfmus[samp_col].loc[sn] for par,samp_col in zip(pars,samp_cols)}
				try:	Rhats   = {par:self.dfmus[f'{par}_Rhats'].loc[sn] for par in pars}
				except:	Rhats   = {par:-1.0 for par in pars}
				postplot = POSTERIOR_PLOTTER(samples, pars, parlabels, bounds, Rhats, plotting_parameters)
				Summary_Strs = postplot.corner_plot(verbose=verbose)#Table Summary
			except:
				Summary_Strs = {par:f"${self.dfmus[f'{par}s'].loc[sn]:.2f}\pm{self.dfmus[f'{par}_errs'].loc[sn]:.2f}$" for par in pars}
			#Create Line
			line = self.dfmus['SN'].loc[sn].replace('_','\_') + ' & ' + (r"\multirow{%s}{*}{%s}"%(NSN_g,self.dfmus['Galaxy'].loc[sn].astype(str)) if new_g else '')
			for par in all_PARS:
				if par in pars:
					line += ' & ' + Summary_Strs[par]
				elif par not in pars and par not in pPARS:
					if par in ['zcmb_hats','zHD_hats']:
						line += ' & ' + (r"\multirow{%s}{*}{%s}"%(NSN_g,str(round(self.dfmus[par].loc[sn],6))) if new_g else '')#int(-np.log10(self.dfmus['zhelio_errs'].loc[sn]))))
					elif par in ['cosmo_sample']:
						mark  = {True:"\\cmark",False:"\\xmark"}[self.dfmus[par].loc[sn]]
						line += ' & ' + (r"\multirow{%s}{*}{%s}"%(NSN_g,{True:"\\cmark",False:"\\xmark"}[self.dfmus[par].loc[sn]]) if new_g else '')
					else:
						line += ' & ' + (r"\multirow{%s}{*}{%s}"%(NSN_g,self.dfmus[par].astype(str).loc[sn]) if new_g else '')
			line +=  ' \\\\ '
			lines.append(line)
		if returner:
			return lines
		else:
			print ('\n'.join(lines))



	def __init__(self,dfmus,samplename='full',sigma0=0.1,sigmapec=250,eta_sigmaRel_input=None,use_external_distances=False,zcosmo='zHD',rootpath='./',FS=18,verbose=True,fiducial_cosmology={'H0':73.24,'Om0':0.28}):
		"""
		Initialisation

		Parameters
		----------
		dfmus : pandas df
			dataframe of individual siblings distance estimates with columns Galaxy, SN, mus, mu_errs

		samplename : str (optional; default='full')
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

		zcosmo : str (optional; default='zHD')
			choice of whether cosmology redshift is flow-corrected zHD_hats or un-flow-corrected zcmb_hats
			'zHD' or 'zcmb'

		rootpath : str
			path/to/sigmaRel/rootpath

		FS : float (optional; default=18)
			fontsize for plots

		verbose: bool (optional; default=True)
			if True, print input dataframe

		fiducial_cosmology : dict (optional; default={'H0':73.24,'Om0':0.28})
			cosmo parameters used in astropy FlatLambdaCDM for creating external redshift-based distances
		"""
		#Data
		if str(type(dfmus))=="<class 'dict'>": dfmus = pd.DataFrame(dfmus)
		self.dfmus      = dfmus
		self.samplename = samplename
		self.dfmus.sort_values(['Galaxy','SN'],ascending=True,inplace=True)
		self.verbose    = verbose

		try:#Reorder and re-label galaxies. Order by redshift, and within this, by zhelio error
			ordered_galaxies = self.dfmus.groupby('Galaxy')[['zhelio_hats','zhelio_errs']].agg('mean').sort_values(by=['zhelio_hats','zhelio_errs']).index
			new_galaxies     = dict(zip(ordered_galaxies.values,np.arange(1,self.dfmus.Galaxy.max()+1,dtype=int)))
			self.dfmus       = self.dfmus.set_index('Galaxy').loc[ordered_galaxies].reset_index()
			self.dfmus['Galaxy'] = self.dfmus['Galaxy'].map(new_galaxies)
		except:
			pass

		self.infer_pars()
		self.fiducial_cosmology = fiducial_cosmology
		self.get_redshift_distances()#Create ['muext_zcmb_hats','muext_zHD_hats'] columns
		self.get_cosmo_subsample()#Create cosmo_sample column
		self.og_dfmus = copy.deepcopy(self.dfmus) ; self.og_samplename = copy.deepcopy(self.samplename)

		#Input Posterior Parameters
		self.master_parnames  = ['mu','AV','theta','RV']
		self.master_parlabels = ['$\\mu$ (mag)','$A_V$ (mag)','$\\theta$','$\\R_V$']
		self.master_bounds    = [[None,None],[0,None],[None,None],[None,None]]

		#Model
		self.sigma0                 = sigma0
		self.sigmapec               = sigmapec
		self.eta_sigmaRel_input     = eta_sigmaRel_input
		self.use_external_distances = use_external_distances
		self.zcosmo			        = zcosmo

		#Paths
		self.packagepath  = os.path.dirname(os.path.abspath(__file__))#The package path
		self.rootpath     = rootpath  #The project rootpath
		self.modelpath    = self.rootpath  + 'model_files/' #Where model_files will be stored (copied to from package)
		self.stanpath     = self.modelpath + 'stan_files/MultiGalFiles/'
		self.productpath  = self.rootpath  + 'products/multigal/'
		self.plotpath     = self.rootpath  + 'plots/multi_galaxy_plots/'
		self.create_paths()
		try:
			shutil.copytree(os.path.join(self.packagepath,'stan_files'), self.modelpath+'stan_files')#Copy stan_files from packagepath to modelpath (if local dev. these are the same)
		except:
			print (f"Tried copying stan_files folder from :{os.path.join(self.packagepath,'stan_files')} to {self.modelpath+'stan_files'}")
			print ("But the latter folder already exists.")

		#Posterior Configuration
		self.n_warmup   = 1000
		self.n_sampling = 5000
		self.n_chains   = 4
		self.n_thin     = 1000

		#Other
		self.FS = FS
		self.c_light = 299792458
		if self.verbose:
			print (self.dfmus[['SN','Galaxy']+self.parcols+self.zhelio_cols+self.zcosmo_cols+self.mucosmo_cols])

	def trim_sample(self,key='cosmo',bool_column=None):
		"""
		Trim Sample

		Simple method to take dfmus and cut based on some bool column
		Default is to cut to the cosmology sub-sample, and the bool column is pre-computed already

		Parameters
		----------
		key: str (optional; default='cosmo')
			sample key used for labelling/saving/loading plots/chains

		bool_column: pd.Series (optional; default=None)
			used to filter dfmus[bool_column]

		End Product(s)
		----------
		assigns trimmed self.dfmus, and new self.samplename
		"""
		if key=='cosmo':
			try:
				assert(self.samplename==self.og_samplename)
				self.dfmus = self.dfmus[self.dfmus.cosmo_sample]
				self.samplename = 'cosmo'
			except:
				raise Exception('If trying to trim to cosmology sub-sample, ensure original sample is loaded first by running .restore_sample()')
		else:
			self.dfmus = self.dfmus[bool_column]
			self.samplename = key

	def restore_sample(self):
		"""
		Restore Sample

		Restores dfmus to full sample (after e.g. cutting to the cosmology sub-sample)
		"""
		if self.samplename!=self.og_samplename:
			self.dfmus      = copy.deepcopy(self.og_dfmus)
			self.samplename = copy.deepcopy(self.og_samplename)
		else:
			pd.testing.assert_frame_equal(self.dfmus,self.og_dfmus)#Assert they are the same
			print ('Full sample already in use')

	def update_attributes(self,other_class,attributes_to_add = ['modelkey','sigma0','sigmapec','sigmaRel_input','eta_sigmaRel_input','use_external_distances','zcosmo']):
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
		parnames   = ['sigma0','sigmapec','rho','sigmaRel','sigmaCommon','rel_rat','com_rat','rel_rat2','com_rat2']
		dfparnames = ['sigma0','sigmapec','rho','sigmaRel','sigmaCommon','rel_rat','com_rat','rel_rat2','com_rat2']
		parlabels  = ['$\\sigma_{0}$ (mag)','$\\sigma_{\\rm{pec}}$ (km$\,$s$^{-1}$)','$\\rho$','$\\sigma_{\\rm{Rel}}$ (mag)','$\\sigma_{\\rm{Common}}$ (mag)','$\\sigma_{\\rm{Rel}}/\\sigma_{0}$','$\\sigma_{\\rm{Common}}/\\sigma_{0}$','$\\sigma^2_{\\rm{Rel}}/\\sigma^2_{0}$','$\\sigma^2_{\\rm{Common}}/\\sigma^2_{0}$']
		bounds     = [[0,1],[0,self.c_light],[0,1],[0,None],[0,None],[0,1],[0,1],[0,1],[0,1]]

		#Filter on pars
		PARS  = pd.DataFrame(data=dict(zip(['dfparnames','parlabels','bounds'],[dfparnames,parlabels,bounds])),index=parnames)
		PARS = PARS.loc[pars]

		#Set class attributes
		self.parnames   = pars
		self.dfparnames = PARS['dfparnames'].values
		self.parlabels  = PARS['parlabels'].values
		self.bounds     = PARS['bounds'].values

		if len(pars)==1 and pars[0]=='sigmaRel':
			self.bounds = [[0,self.sigma0]]
			print (f'Updating sigmaRel bounds to {self.bounds} seeing as sigma0/sigmapec are fixed')

	def sigmaRel_sampler(self, sigma0=None, sigmapec=None, eta_sigmaRel_input=None, use_external_distances=None, zmarg=False, alt_prior=False, overwrite=True, blind=False, zcosmo=None, alpha_zhel=False, Rhat_threshold=np.inf, Ntrials=1):
		"""
		sigmaRel Sampler

		Core method, used to take individual siblings distance estimates to multiple galaxies, and compute posteriors on sigmaRel
		Options to:
			- fix or free sigma0
			- fix or free sigmapec
			- fix or free correlation coefficient rho = sigmalRel/sigma0
			- include external distance constraints
			- marginalise over z_g parameters
			- choose different sigma_int priorss, default is sigma0~Uninformative and sigmaRel~U(0,sigma0); alternative is sigmaRel, sigmaCommon~Uninformative


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

		zmarg : bool (optional; default=False)
			if False, use LCDM distances and Gaussian uncertainty approximation in distance
			if True, use z_g data and draw Gaussian z_g parameters

		alt_prior : bool (optional; default=False)
			two choices of prior are [sigma0~Prior and sigmaRel~U(0,sigma0)] OR [sigmaRel~Prior; sigmaCommon~Prior]
			alt_prior=False is former, alt_prior=True is latter

		overwrite : bool (optional; default=True)
			option to overwrite previously saved .pkl posterior samples

		blind : bool (optional; default=False)
			option to mask posterior results

		zcosmo : str or None (optional; default=None)
			choice of whether cosmology redshift is flow-corrected zHD_hats or un-flow-corrected zcmb_hats
			if None, set to self.zcosmo
			if str, set as either 'zHD' or 'zcosmo'

		alpha_zhel : bool (optional; default=False)
			if zmarg is True, then alpha_zhel can be activated. This takes the pre-computed slopes of dmu_phot = alpha*dzhelio and marginalises over this in the z-pipeline

		Rhat_threshold : float (optional; default=np.inf)
			float value if Rhat>Rhat_threshold redo fit with more samples, repeat for up to Ntrials times

		Ntrials : int (optional; default=1)
			no. of times to re-do fit with more samples to match Rhat threshold target

		End Product(s)
		----------
		self.FIT, posterior samples saved in self.productpath
		"""
		if alt_prior==True and sigma0!='free':
			raise Exception('Cannot use alternative prior and while not freeing sigma0')
		if alpha_zhel and not zmarg:
			raise Exception('Cannot activate alpha_zhel mode with mu-pipeline; please set zmarg=True')
		#Initialise model with choices
		modelloader = ModelLoader(sigma0, sigmapec, eta_sigmaRel_input, use_external_distances, zmarg, alt_prior, zcosmo, alpha_zhel, self)
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
			stan_init   = modelloader.get_stan_init(dfmus, self.productpath, self.n_chains)
			self.pars   = modelloader.get_pars()

			#Initialise model
			model       = CmdStanModel(stan_file=self.stanpath+'current_model.stan')
			#Fit model
			for itrial in range(Ntrials):
				BREAK = True
				fit         = model.sample(data=stan_data,chains=self.n_chains, iter_sampling=self.n_sampling*(1+itrial), iter_warmup = self.n_warmup, inits=stan_init, seed=42+itrial)
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

				#Check for high Rhat values
				for par in self.pars:
					if fitsummary.loc[par]['r_hat']>Rhat_threshold:
						BREAK = False
				if BREAK:	break
				else:		print (f'Completed {itrial+1}/{Ntrials}; Repeating fit with more samples because of Rhats:',{par:fitsummary.loc[par]['r_hat'] for par in self.pars})

			#Save samples
			FIT = dict(zip(['data','summary','chains','modelloader'],[stan_data,fitsummary,df,modelloader]))
			with open(self.productpath+f"FIT{self.samplename}{self.modelkey}.pkl",'wb') as f:
				pickle.dump(FIT,f)

			#Print posterior summaries
			print (f"Fit Completed; Summary is:")
			for col in self.pars:
				print ('###'*10)
				print (f'{col}, median, std, 16, 84 // 68%, 95%')
				if not blind:
					print (df[col].median().round(3),df[col].std().round(3),df[col].quantile(0.16).round(3),df[col].quantile(0.84).round(3))
					print (df[col].quantile(0.68).round(3),df[col].quantile(0.95).round(3))
				print ('Rhat:', fitsummary.loc[col]['r_hat'])

		else:#Else load up
			with open(self.productpath+f"FIT{self.samplename}{self.modelkey}.pkl",'rb') as f:
				FIT = pickle.load(f)
			self.pars   = modelloader.get_pars()#For plotting

		#Print distance summaries
		self.FIT  = FIT

	def add_transformed_params(self,df,fitsummary):
		"""
		Add Transformed Parameters

		Simple method take compute new transformed parameters from original parameter samples

		Parameters
		----------
		df: pandas df
			contains posterior samples

		fitsummary: pandas df
			from arviz, contains summaries including Rhats

		Returns
		----------
		df, fitsummary with updated values
		"""
		#Added Parameters
		added_pars = ['rel_rat','com_rat','rel_rat2','com_rat2']
		try:
			df['rel_rat'] = df['sigmaRel']/df['sigma0']
			df['com_rat'] = df['sigmaCommon']/df['sigma0']
			df['rel_rat2'] = df['sigmaRel'].pow(2)/df['sigma0'].pow(2)
			df['com_rat2'] = df['sigmaCommon'].pow(2)/df['sigma0'].pow(2)
			#Appender
			added_fitsummary = az.summary({x:np.array_split(df[x].values,self.n_chains) for x in added_pars})
			for x in added_pars:
				fitsummary.loc[x,'r_hat'] = added_fitsummary.loc[x]['r_hat']
		except Exception as e:
			print (e)
		return df, fitsummary

	def plot_posterior_samples(self, FS=18,paperstyle=True,quick=True,save=True,show=False, returner=False, blind=False, fig_ax=None, **kwargs):
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

		blind : bool (optional; default=False)
			option to mask posterior results

		fig_ax : None or list (optional; default=None)
			if None, create new figure, else, fig_ax = [postplot,counter,Npanel,line_index,legend_labels],
			counter denotes how many times figure has been used
			Npanel is how many sample files in total
			line_index indicates (where applicable) how many descriptive lines to use in plot
			legend_labels is list of lines describing individual model

		End Product(s)
		----------
		Plot
		"""
		'''#This code is now outdated, thus for now make sure to use plot_posterior_samples after sigmaRel sampler
		#Get pars to plot
		if 'modelkey' not in self.__dict__:
			modelloader = ModelLoader(self.sigma0, self.sigmapec, self.eta_sigmaRel_input, self.use_external_distances, self)
			modelloader.get_model_params()
			modelloader.get_modelkey()
			self.update_attributes(modelloader)
			self.pars   = modelloader.get_pars()
		self.get_parlabels(self.pars)
		#'''

		#Get posterior samples
		if 'FIT' in self.__dict__.keys():
			FIT = self.FIT
		else:
			savekey  = self.samplename+self.modelkey
			filename = self.productpath+f"FIT{savekey}.pkl"
			with open(filename,'rb') as f:
				FIT = pickle.load(f)

		#Get posterior data
		df         = FIT['chains'] ; fitsummary = FIT['summary'] ; stan_data  = FIT['data'] ; modelloader = FIT['modelloader']
		df, fitsummary = self.add_transformed_params(df, fitsummary)
		for parq in [[x.split('filt_')[1],kwargs[x]] for x in kwargs.keys() if  'filt_' in x]:
			par,q = parq[:]
			df = df[(df[par]>=df[par].quantile(q[0])) & (df[par]<=df[par].quantile(q[1]))]

		if 'pars' in kwargs.keys():
			self.get_parlabels(kwargs['pars'])
		else:
			self.get_parlabels(modelloader.get_pars())
		Rhats      = {par:fitsummary.loc[par]['r_hat'] for par in self.dfparnames}
		samples    = {par:np.asarray(df[par].values)   for par in self.dfparnames}
		print ('Rhats:',Rhats)

		#Corner Plot
		self.plotting_parameters = {'FS':FS,'paperstyle':paperstyle,'quick':quick,'save':save,'show':show}
		if fig_ax is None:#Single Plot
			postplot = POSTERIOR_PLOTTER(samples, self.parnames, self.parlabels, self.bounds, Rhats, self.plotting_parameters)
			colour = None if 'colour' not in kwargs.keys() else kwargs['colour'] ; counter = 0; Npanel = 1; line_index = [[None,None]] ; multiplot = False ; postplot.y0 = None
		else:
			multiplot = True
			if fig_ax[0] is None:#Multiplot, and first instance
				postplot = POSTERIOR_PLOTTER(samples, self.parnames, self.parlabels, self.bounds, Rhats, self.plotting_parameters)
			else:#Multiplot, and after first instance
				postplot = fig_ax[0]
				postplot.samples = samples ; postplot.Rhats   = Rhats
				postplot.chains  = [postplot.samples[par] for par in postplot.parnames]
				postplot.choices = self.plotting_parameters
			counter,Npanel,line_index,legend_labels = fig_ax[1:5]
			if type(line_index[0]) is not list: line_index = [line_index]
			colour = f'C{counter + (0 if len(fig_ax)==5 else fig_ax[5])}'
			postplot.lines  = get_Lines(stan_data,self.c_light,modelloader.alt_prior,modelloader.zcosmo,modelloader.alpha_zhel)
			if counter==0 and fig_ax[0] is None:
				postplot.lc = sum([len(postplot.lines[ll[0]:ll[1]]) for ll in line_index])#len(postplot.lines[line_index[0]:line_index[1]])
			postplot.y0  = len(self.parnames)+(Npanel+1*(len(self.parnames)<4))*0.15
			FS += 0 + -2*(len(self.parnames)<4)

		Summary_Strs    = postplot.corner_plot(verbose=not blind,blind=blind,colour=colour,multiplot=False if not multiplot else [counter if legend_labels!=[''] else -1,Npanel])#Table Summary
		if multiplot:#Plot multiplot lines
			dy  = (0.15-0.02*(len(self.parnames)<4))
			yy0 = postplot.y0-0.35+0.06*(len(self.parnames)<4)
			for ticker,line in enumerate(legend_labels):
				pl.annotate(line, xy=(1+1.1*(len(postplot.ax)==1),yy0-dy*(postplot.lc+ticker-1)),xycoords='axes fraction',
							fontsize=FS-4,color=colour,ha='right')
			if counter+1==Npanel:
				ticker = -1
				for ll in line_index:
					for line in postplot.lines[ll[0]:ll[1]]:
						ticker+=1
						pl.annotate(line, xy=(1+1.1*(len(postplot.ax)==1),yy0-dy*(ticker-1)),xycoords='axes fraction',
									fontsize=FS-4,color='black',ha='right')

		savekey         = self.samplename+self.modelkey+'_FullKDE'*bool(not self.plotting_parameters['quick'])+'_NotPaperstyle'*bool(not self.plotting_parameters['paperstyle'])
		save,quick,show = [self.plotting_parameters[x] for x in ['save','quick','show']][:]
		if counter+1==Npanel:
			if counter+1==Npanel:
				LINES = get_Lines(stan_data,self.c_light,modelloader.alt_prior,modelloader.zcosmo,modelloader.alpha_zhel)
				LINES = [LINES[ll[0]:ll[1]] for ll in line_index]
				LINES = [L for LL in LINES for L in LL]
			elif counter+1!=Npanel:
				LINES = []
			finish_corner_plot(postplot.fig,postplot.ax,LINES,save,show,self.plotpath,savekey,colour if not multiplot else 'black',y0=postplot.y0,lines= not multiplot)

		if multiplot and legend_labels!=['']: postplot.lc += len(legend_labels)
		#Return posterior summaries
		if returner:
			if multiplot:
				return Summary_Strs, postplot
			else:
				return Summary_Strs
		else:
			if multiplot:
				return postplot


	def plot_posterior_samples_1D(self, FS=18,paperstyle=True,quick=True,save=True,show=False, returner=False, blind=False, fig_ax=None, **kwargs):
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

		blind : bool (optional; default=False)
			option to mask posterior results

		fig_ax : None or list (optional; default=None)
			if None, create new figure, else, fig_ax = [postplot,counter,Npanel,line_index,legend_labels],
			counter denotes how many times figure has been used
			Npanel is how many sample files in total
			line_index indicates (where applicable) how many descriptive lines to use in plot
			legend_labels is list of lines describing individual model

		End Product(s)
		----------
		Plot
		"""
		#Get posterior samples
		if 'FIT' in self.__dict__.keys():
			FIT = self.FIT
		else:
			savekey  = self.samplename+self.modelkey
			filename = self.productpath+f"FIT{savekey}.pkl"
			with open(filename,'rb') as f:
				FIT = pickle.load(f)

		#Get posterior data
		df         = FIT['chains'] ; fitsummary = FIT['summary'] ; stan_data  = FIT['data'] ; modelloader = FIT['modelloader']
		df, fitsummary = self.add_transformed_params(df, fitsummary)
		if fig_ax is not None and 'pars' in kwargs.keys():
			self.get_parlabels(kwargs['pars'])
		else:
			self.get_parlabels(modelloader.get_pars())
		Rhats      = {par:fitsummary.loc[par]['r_hat'] for par in self.dfparnames}
		samples    = {par:np.asarray(df[par].values)   for par in self.dfparnames}
		print ('Rhats:',Rhats)

		#Corner Plot
		self.plotting_parameters = {'FS':FS,'paperstyle':paperstyle,'quick':quick,'save':save,'show':show}
		if fig_ax is None:#Single Plot
			postplot = POSTERIOR_PLOTTER(samples, self.parnames, self.parlabels, self.bounds, Rhats, self.plotting_parameters)
			colour = None ; counter = 0; Npanel = 1; line_index = [[None,None]] ; multiplot = False ; postplot.y0 = None
		else:
			multiplot = True
			if fig_ax[0] is None:#Multiplot, and first instance
				postplot = POSTERIOR_PLOTTER(samples, self.parnames, self.parlabels, self.bounds, Rhats, self.plotting_parameters)
			else:#Multiplot, and after first instance
				postplot = fig_ax[0]
				postplot.samples = samples ; postplot.Rhats   = Rhats
				postplot.chains  = [postplot.samples[par] for par in postplot.parnames]
				postplot.choices = self.plotting_parameters
			counter,Npanel,line_index,legend_labels = fig_ax[1:5]
			if type(line_index[0]) is not list: line_index = [line_index]
			colour = f'C{counter + (0 if len(fig_ax)==5 else fig_ax[5])}'
			postplot.lines  = get_Lines(stan_data,self.c_light,modelloader.alt_prior,modelloader.zcosmo,modelloader.alpha_zhel)
			if counter==0 and fig_ax[0] is None:
				postplot.lc = sum([len(postplot.lines[ll[0]:ll[1]]) for ll in line_index])#len(postplot.lines[line_index[0]:line_index[1]])
			postplot.y0  = 1.3#len(self.parnames)+(Npanel+1*(len(self.parnames)<4))*0.15
			FS += 0 + -2*(len(self.parnames)<4)

		Summary_Strs    = postplot.plot_1Drow(verbose=not blind,blind=blind,colour=colour,multiplot=False if not multiplot else [counter if legend_labels!=[''] else -1,Npanel])#Table Summary
		if multiplot and 'lines' in kwargs:
			x0  = 1.15
			if kwargs['lines']:
				dy  = (0.15-0.02*(len(self.parnames)<4))
				yy0 = postplot.y0-0.35+0.06*(len(self.parnames)<4)
				for ticker,line in enumerate(legend_labels):
					pl.annotate(line, xy=(x0+1.1*(len(postplot.ax)==1),yy0-dy*(postplot.lc+ticker-1)),xycoords='axes fraction',
								fontsize=FS-4,color=colour,ha='right')
				if counter+1==Npanel:
					pl.annotate(r"sigmaRel_computer",     xy=(x0+1.1*(len(postplot.ax)==1),postplot.y0-0.025),xycoords='axes fraction',fontsize=18,color='black',weight='bold',ha='right',fontname='Courier New')
					ticker = -1
					for ll in line_index:
						for line in postplot.lines[ll[0]:ll[1]]:
							ticker += 1
							pl.annotate(line, xy=(x0+1.1*(len(postplot.ax)==1),yy0-dy*(ticker-1)),xycoords='axes fraction',
										fontsize=FS-4,color='black',ha='right')
			elif not kwargs['lines']:
				pl.annotate(r"sigmaRel_computer",     xy=(x0+1.1*(len(postplot.ax)==1),postplot.y0-0.025),xycoords='axes fraction',fontsize=18,color='white',weight='bold',ha='right',fontname='Courier New')



		if 'savekey' not in kwargs:
			savekey = self.samplename+self.modelkey+'_FullKDE'*bool(not self.plotting_parameters['quick'])+'_NotPaperstyle'*bool(not self.plotting_parameters['paperstyle'])
		else:
			savekey = kwargs['savekey']
		save,quick,show = [self.plotting_parameters[x] for x in ['save','quick','show']][:]
		if counter+1==Npanel:
			if multiplot:
				for col in range(len(self.parnames)):
					postplot.ax[0,col].set_ylim([0,postplot.ax[0,col].get_ylim()[1]])
			if counter+1==Npanel:
				LINES = get_Lines(stan_data,self.c_light,modelloader.alt_prior,modelloader.zcosmo,modelloader.alpha_zhel)
				LINES = [LINES[ll[0]:ll[1]] for ll in line_index]
				LINES = [L for LL in LINES for L in LL]
			elif counter+1!=Npanel:
				LINES = []
			finish_corner_plot(postplot.fig,postplot.ax,LINES,save,show,self.plotpath,savekey,None if not multiplot else 'black',y0=postplot.y0,lines= not multiplot,oneD=True)

		if multiplot and legend_labels!=['']: postplot.lc += len(legend_labels)
		#Return posterior summaries
		if returner:
			if multiplot:
				return Summary_Strs, postplot
			else:
				return Summary_Strs
		else:
			if multiplot:
				return postplot



	def compute_analytic_multi_gal_sigmaRel_posterior(self,PAR='mu',prior_upper_bounds=[1.0],alpha_zhel=False,show=False,save=True,blind=False,fig_ax=None,prior=None):
		"""
		Compute Analytic Multi Galaxy sigmaRel Posterior

		Method to compute analytic sigmaRel posterior by multiplying the single-galaxy likelihoods by the prior

		Parameters
		----------
		PAR : str (optional; default='mu')
			parameter to constrain dispersion for

		prior_upper_bounds : list (optional; default=[1.0])
			choices of sigmaRel prior upper bound

		alpha_zhel : bool (optional; default=False)
			When true, this takes the pre-computed slopes of dmu_phot = alpha*dzhelio and marginalises over this (at present valid only for Nsib=2 galaxies)

		show : bool (optional; default=False)
			bool to show plots or not

		save : bool (optional; default=True)
			bool to save plots or not

		blind : bool (optional; default=False)
			if True, blind sigmaRel plot axes and posterior summaries

		fig_ax : None or list (optional; default=None)
			if None, create new figure, else, fig_ax = [postplot,counter,Npanel,line_index,legend_labels],
			counter denotes how many times figure has been used
			Npanel is how many sample files in total
			line_index indicates (where applicable) how many descriptive lines to use in plot
			legend_labels is list of lines describing individual model

		End Product(s)
		----------
		Plot of multi-galaxy sigmaRel posterior

		self.total_posteriors: dict
			key,value pairs are the sigmaRel prior upper bound, and the posterior

		self.sigRs_store: dict
		 	same as self.total_posteriors, but the sigR prior grid
		"""
		def zhel_modify(dfgal):#Takes Nsiblings=2 galaxy, selects the sibling with the larger fitting error, and adds the muphot_zhelio uncorrelated error component
			#print ('Before:', dfgal[['alpha_mu_z',f'{PAR}_errs']])
			if dfgal['alpha_mu_z'].nunique()>1 and dfgal['alpha_mu_z'].values[0]!=0:#if this siblings galaxy has a large zhelio error and thus has alpha estimated
				if dfgal.shape[0]!=2:
					raise Exception('Simple analytic trick for incorporating muphot-zhelio errors only works for Nsiblings=2 per galaxy; easiest workaround is to sample posterior')
				else:
					delta_alpha = abs(dfgal['alpha_mu_z'].iloc[0]-dfgal['alpha_mu_z'].iloc[1])
					imod_sib    = 0 if dfgal['alpha_mu_z'].iloc[0]>dfgal['alpha_mu_z'].iloc[1] else 1#The sibling with the larger alpha
					new_sigmafit = (dfgal[f'{PAR}_errs'].iloc[imod_sib]**2 + (delta_alpha*dfgal['zhelio_errs'].iloc[imod_sib])**2 )**0.5
					dfgal.loc[dfgal.index.values[imod_sib],f'{PAR}_errs'] = new_sigmafit
			#print ('After:', dfgal[['alpha_mu_z',f'{PAR}_errs']])
			return dfgal

		#List of prior upper bounds to loop over
		self.prior_upper_bounds = prior_upper_bounds

		#Initialise posteriors
		total_posteriors = {p:1 for p in self.prior_upper_bounds} ; sigRs_store = {}
		#For each galaxy, compute sigmaRel likelihood
		for g,gal in enumerate(self.dfmus['Galaxy'].unique()):
			dfgal  = self.dfmus[self.dfmus['Galaxy']==gal].copy()
			if alpha_zhel: dfgal = zhel_modify(dfgal)														#Use dummy value for sigma0
			sibgal = siblings_galaxy(dfgal[f'{PAR}s'].values,dfgal[f'{PAR}_errs'].values,dfgal['SN'].values,gal,sigma0=0.1,prior_upper_bounds=self.prior_upper_bounds,rootpath=self.rootpath)
			sibgal.get_sigmaRel_posteriors()
			for p in self.prior_upper_bounds:
				total_posteriors[p] *= sibgal.posteriors[p]*p #Multiply likelihoods, so divide out prior for each galaxy
				if g==0:
					total_posteriors[p] *= 1/p#Prior only appears once
					sigRs_store[p] = sibgal.sigRs_store[p]

		#Add in custom hyperprior
		def get_sigR_prior(prior,sigRs):
			if prior in ['uniform',None]:
				return np.ones(len(sigRs))
			elif prior in ['p2']:
				return sigRs
			elif prior in ['jeffreys']:
				return 1/sigRs
			elif prior[0]=='p':
				#return np.power(sigRs,float(prior[1:])-1)#e.g. prior='p2' is sigRs^{1}
				return np.power(sigRs,float(prior[1:])-1)#*np.power(1-np.square(sigRs),float(prior[1:])/2-1)#e.g. prior='p2' is sigRs^{1}
			else:
				raise Exception('Selected prior not defined')

		if prior is not None:
			assert(type(prior)==str)
			for p in self.prior_upper_bounds:
				total_posteriors[p] =  copy.deepcopy(total_posteriors[p])*get_sigR_prior(prior,sibgal.sigRs_store[p])

		#Plot posteriors
		if self.verbose:
			#Use single-galaxy class for plotting
			for p in self.prior_upper_bounds:
				sibgal.posteriors[p]  = total_posteriors[p]
				sibgal.sigRs_store[p] = sigRs_store[p]

				sibgal.posterior = total_posteriors[p]
				print ('50%-16%:',sibgal.get_quantile(0.5,return_index=False)-sibgal.get_quantile(0.16,return_index=False))
				print ('50%:',sibgal.get_quantile(0.5,return_index=False))
				print ('84%-50%:',sibgal.get_quantile(0.84,return_index=False)-sibgal.get_quantile(0.5,return_index=False))
				print ('95%:',sibgal.get_quantile(0.95,return_index=False))

			sibgal.show     = show
			sibgal.save     = save
			sibgal.galname  = self.samplename
			sibgal.plotpath = self.plotpath
			sibgal.plot_sigmaRel_posteriors(xupperlim='adaptive',blind=blind, fig_ax=fig_ax)#Assigns fig_ax attribute inside method

		#Store posteriors as attribute
		self.total_posteriors = total_posteriors
		self.sigRs_store      = sigRs_store
		if sibgal.fig_ax is not None:
			return sibgal.fig_ax[0], sibgal.fig_ax[1], sibgal.fig_ax[2] #fig, ax, kmax


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

	def get_dxgs(self,Sg,ss,g_or_z):
		"""
		Get dxgs

		Simple method for SN labelling coordinates in the plot_all_distance method
		If first SN, shift to left, if last SN shift to right, else, update with rule

		Parameters
		----------
		Sg,ss : floats
			number of SNe in galaxy, and 0-index of current sn

		g_or_z : str
			either g or z, for photometric or redshift mode
		"""
		if g_or_z == 'g':
			dd = 0.1
		else:
			dd = None

		if ss==0:
			dx = -dd; ha='right'
		elif ss==Sg-1:
			dx = dd; ha='left'
		else:
			dx = dd; ha='left'
		return dx, ha

	def plot_delta_HR(self,save=True,show=False):
		'''
		Plot Delta HR

		Indicates how HR changes from adding flow corrections

		Parameters
		----------
		show, save: bools (optional; default=False,True)
			whether to show/save plot

		End Product(s)
		----------
		Figure of |HR|CMB - |HR|HD
		'''
		#Initialise
		pl.figure(figsize=(9.6,7.2))
		markersize=14;capsize=5;alpha=0.75;elw=3;mew=3
		Ng = self.dfmus['Galaxy'].unique()
		colours = [f'C{s%10}' for s in range(self.dfmus['Galaxy'].nunique())]
		markers = ['o','s','p','^','x','P','d','*']
		markers = [markers[int(ig%len(markers))] for ig in range(Ng.shape[0])]

		#Get Hubble Residuals
		mini_d   = 0.0035
		cosmo    = FlatLambdaCDM(H0=73.24,Om0=0.28)
		zwords = ['zcmb_hats','zHD_hats']
		HR_collection = {zword:{} for zword in zwords}
		for zword in zwords:
			z_hats = self.dfmus.groupby('Galaxy',sort=False)[zword].agg('mean').values
			for igal,gal in enumerate(Ng):
				dfgal  = self.dfmus[self.dfmus['Galaxy']==gal].copy()
				muext  = dfgal[f'muext_{zword}'].values[0]#cosmo.distmod(z_hats[igal]).value
				for mu,ss in zip(dfgal['mus'].values,np.arange(dfgal.shape[0])):
					HR_collection[zword][f'{igal}_{ss}'] = mu-muext

		#Plot change in HR
		zHDs   = self.dfmus.groupby('Galaxy',sort=False)['zHD_hats'].agg('mean').values
		for igal,gal in enumerate(Ng):
			dfgal  = self.dfmus[self.dfmus['Galaxy']==gal]
			Sg     = dfgal.shape[0]
			#xgs    = np.array([mini_d*(ss-(Sg-1)/2) for ss in range(Sg)]) + zHDs[igal]#Plot against zHD
			xgs    = np.array([np.exp(0.4*(ss-(Sg-1)/2)) for ss in range(Sg)])*dfgal['zhelio_errs'].mean()
			for ss in range(Sg):
				HR_cmb = abs(HR_collection['zcmb_hats'][f'{igal}_{ss}'])
				HR_hd  = abs(HR_collection['zHD_hats'][f'{igal}_{ss}'])
				pl.errorbar(xgs[ss],HR_cmb-HR_hd,color=colours[igal],marker=markers[igal],markersize=markersize,alpha=alpha,elinewidth=elw,markeredgewidth=mew,linestyle='none',capsize=capsize)

		pl.ylabel(r"$|$HR$_{\rm{CMB}}|$-$|$HR$_{\rm{HD}}|$",fontsize=self.FS)
		#pl.xlabel(r"$z_{\rm{HD}}$",fontsize=self.FS)#Plot against zHD
		pl.xlabel(r"Error in $z_{\rm{Helio}}$",fontsize=self.FS) ; pl.xscale('log')
		pl.gca().set_xlim([0,self.dfmus['zHD_hats'].max()+0.01])
		pl.plot(pl.gca().get_xlim(),[0,0],color='black',linewidth=0.2)
		pl.tight_layout()
		pl.tick_params(labelsize=self.FS)
		if save:
			pl.savefig(f"{self.plotpath}HRChange.pdf", bbox_inches="tight")
		if show:
			pl.show()

	def plot_parameters(self, PAR='mu',colours=None, markers=None, g_or_z = 'g',subtract_g_mean=None,zword='zHD_hats',
							show=False, save=True,
							markersize=14,capsize=5,alpha=0.9,elw=3,mew=3, plot_full_errors=False,plot_sigma0=0.094,plot_sigmapec = 250,
							text_index = 3, annotate_mode = 'legend',
							args_legend={'loc':'upper left','ncols':2,'bbox_to_anchor':(1,1.02)},**kwargs):
		"""
		Plot All Distances

		Method to plot up all photometric distances for all siblings galaxies. Can be used with or without external distance constraints (controlled by g_or_z)

		Parameters
		----------
		PAR: str or list (optional; default='mu')
			if str, plots parameter, if list, plots multiple parameters

		colours: list of str (optional; default=None)
			colours used for each Galaxy

		markers: list of str (optional; default=None)
			markers used for each Galaxy

		g_or_z: str (optional; default='g')
			defines whether to plot only photometric distances by GalaxyID ('g')
			or to include redshift-based distances and plot by redshift

		subtract_g_mean: None (optional; default=None)
			if None, becomes True for PAR=='mu' or False for PAR=='theta','AV','RV'

		zword : str (optional; default='zHD_hats')
			option to plot zHD or could choose e.g. zcmb_hats

		show, save: bools (optional; default=False,True)
			whether to show/save plot

		markersize,capsize,alpha,elw,mew,plot_full_errors,plot_sigma0,plot_sigmapec: float,float,bool,foat (optional;default=14,5,0.9,3,3,False,0.094)
			size of markers and cap., alpha, Can include additional overlay where mu errors include a sigma0 term (default is 0.094~mag from W22), controlled by plot_full_errors bool, sigmapec envelope for HRs

		text_index, annotate_mode: int, str (optional; default=3, 'legend' or 'text')
			for example if 'ZTF18abcxyz', specifying 3 returns label '18abcxyz'
			if legend annotate, put SN names in legend, otherwise, put SN name next to data points

		args_legend: dict (optional; default={'loc':'upper center','ncols':3})
			legend arguments

		End Product(s)
		----------
		Plot of siblings distance estimate across all siblings galaxies
		"""
		multiplot = False if type(PAR)==str else True
		PARS = copy.deepcopy(PAR) if multiplot else [PAR]
		if multiplot:
			args_legend = copy.deepcopy(args_legend)
			args_legend['loc'] = 'center left'
			args_legend['bbox_to_anchor'] = (1,0.5)
			args_legend['ncols'] = 1
		FAC = 3
		self.FS += len(PARS)*FAC

		if [pp for pp in ['AV','RV'] if pp in PARS]!=[]:#To get 68,95 intervals when required
			PPARS = [PAR] if not multiplot else [pp for pp in ['AV','RV'] if pp in PARS]
			lines = self.print_table(PARS=PPARS,returner=True)
			lines = [ll.split('\\\\')[0] for ll in lines if ll!='\\midrule']
			dfsummaries = pd.DataFrame([re.sub(r'\s+','',ll).split('&') for ll in lines],columns=['SN','GalaxyID']+PPARS)
			dfsummaries['SN'] = dfsummaries['SN'].apply(lambda x: x.replace('\_','_'))
			dfsummaries[[f'{pp}bunched' for pp in PPARS]] = dfsummaries[PPARS].apply(lambda x: ['\pm' not in xi for xi in x],axis=1,result_type='expand')
			for pp in PPARS:
				dfsummaries.loc[dfsummaries[f'{pp}bunched'],f'{pp}68'] = dfsummaries.loc[dfsummaries[f'{pp}bunched'],pp].apply(lambda x: x.split('(')[0].split('$')[1][1:])
				dfsummaries.loc[dfsummaries[f'{pp}bunched'],f'{pp}95'] = dfsummaries.loc[dfsummaries[f'{pp}bunched'],pp].apply(lambda x: x.split('(')[1].split(')')[0])
				dfsummaries.loc[dfsummaries[f'{pp}bunched'],f'{pp}bunchindex'] = dfsummaries.loc[dfsummaries[f'{pp}bunched'],pp].apply(lambda x: {'<':0,'>':1}[x[1]])

		#Create plot
		if multiplot:
			fig,ax = pl.subplots(len(PARS),1,figsize=(9.6*(1+0.25*args_legend['ncols']*(annotate_mode=='legend')+0.5*(annotate_mode is None)),6*len(PARS)),sharex='col',sharey=False,squeeze=False)
		else:
			fig,ax = pl.subplots(1,1,figsize=(9.6*(1+0.25*args_legend['ncols']*(annotate_mode=='legend')+0.5*(annotate_mode is None)),7.2),sharex='col',sharey=False,squeeze=False)
			iax=0

		#Get colours markers
		Ng = self.dfmus['Galaxy'].unique()
		if colours is None:
			colours = [f'C{s%10}' for s in range(self.dfmus['Galaxy'].nunique())]
		if markers is None:
			markers = ['o','s','p','^','x','P','d','*']
			markers = [markers[int(ig%len(markers))] for ig in range(Ng.shape[0])]

		#Define xgrid depending on whether mode is photometric distances only or additionally include redshift distances
		if g_or_z=='g':
			x_coords = np.arange(Ng.shape[0])
			mini_d=0.3
		elif g_or_z=='z':
			x_coords = self.dfmus.groupby('Galaxy',sort=False)[zword].agg('mean').values
			mini_d=0.0035
			cosmo  = FlatLambdaCDM(H0=73.24,Om0=0.28)
			#print(x_coords)
			#err=1/0
		else:
			raise Exception(f"Require g_or_z in [g,z], but got {g_or_z}")

		def get_muext_err(sigz,zbar):
			return 5*sigz/(np.log(10)*zbar)

		for iax,PAR in enumerate(PARS):
			subtract_mean = {'mu':True,'theta':False,'AV':False,'RV':False,'etaAV':False}[PAR] if subtract_g_mean is None else subtract_g_mean
			if PAR=='mu':		ylabel = r"$\hat{\mu}_s - \overline{\hat{\mu}_s}$ (mag)" if subtract_mean else r"$\hat{\mu}_s$ (mag)"	;	pword = 'Distance'
			if PAR=='AV':		ylabel = r"$\hat{A}^s_V - \overline{\hat{A}^s_V}$ (mag)" if subtract_mean else r"$\hat{A}^s_V$ (mag)"	;	pword = 'Dust Extinction'
			if PAR=='RV':		ylabel = r"$\hat{R}^s_V - \overline{\hat{R}^s_V}$" 		 if subtract_mean else r"$\hat{R}^s_V$"			;	pword = 'Dust Law Shape'
			if PAR=='theta':	ylabel = r"$\hat{\theta}_s - \overline{\hat{\theta}_s}$" if subtract_mean else r"$\hat{\theta}_s$"		;	pword = 'Light Curve Shape'
			if PAR=='etaAV':	ylabel = r"$\hat{\eta}^s_{A_V} - \overline{\hat{\eta}^s_{A_V}}$ (mag)" if subtract_mean else r"$\hat{\eta}^s_{A_V}$ (mag)"	;	pword = 'Repar. Dust Extinction'

			if iax==0:
				if g_or_z=='z' and not multiplot and PAR=='mu':
					if zword=='zcmb_hats':
						fig.axes[0].set_title(f"Hubble Diagram",weight='bold',fontsize=self.FS+3)
					else:
						pass
				else:
					fig.axes[0].set_title(f"Individual Siblings {pword if not multiplot else 'Parameter'} Estimates",weight='bold',fontsize=self.FS+1)

			if 'include_std' in kwargs and kwargs['include_std'] and g_or_z=='z' and PAR=='mu':
				HRs = [mm[0]-mm[1] for igal,gal in enumerate(Ng) for mm in self.dfmus[self.dfmus['Galaxy']==gal][[f'{PAR}s',f'muext_{zword}']].values]
				fig.axes[iax].annotate(r'Std. Dev. = %s$\,$mag'%(round(np.std(HRs),3)),xy=(0.1,0.1),xycoords='axes fraction',weight='bold',fontsize=self.FS+2,ha='left')
			#For each galaxy, plot distances
			for igal,gal in enumerate(Ng):
				dfgal  = self.dfmus[self.dfmus['Galaxy']==gal]
				mus,muerrs,SNe,Sg = dfgal[f'{PAR}s'].values,dfgal[f'{PAR}_errs'].values,dfgal['SN'].values,dfgal.shape[0]
				if PAR=='mu':	fullerrors = np.array([plot_sigma0**2 + err**2 for err in muerrs])**0.5
				xgs   = np.array([mini_d*(ss-(Sg-1)/2) for ss in range(Sg)]) + x_coords[igal]
				mubar = np.average(mus) if g_or_z=='g' or PAR!='mu' else dfgal[f'muext_{zword}'].values[0]#cosmo.distmod(x_coords[igal]).value
				#Add sigmaext error for HRs
				if g_or_z=='z' and PAR=='mu':
					mu_full_errs = np.array([get_muext_err(zhelerr,x_coords[igal])**2+muerr**2 for muerr,zhelerr in zip(muerrs,dfgal['zhelio_errs'].values)])**0.5

				#For each SN
				for mu,err,ss in zip(mus,muerrs,np.arange(Sg)):
					#Choice of labelling for legend/SNe
					lab = '\n'.join([sn[text_index:] for sn in SNe]) #if g_or_z=='g':#elif g_or_z=='z':	lab = gal
					lab = lab if ss==0 and annotate_mode=='legend' else None
					#Plot points (with optional legend labels)
					if PAR in ['AV','RV'] and dfsummaries[dfsummaries['SN']==SNe[ss]][f'{PAR}bunched'].values[0]:
						bunchindex = int(dfsummaries[dfsummaries['SN']==SNe[ss]][f'{PAR}bunchindex'].values[0])
						point = self.master_bounds[self.master_parnames.index(PAR)][bunchindex]
						e68 = float(dfsummaries[dfsummaries['SN']==SNe[ss]][f'{PAR}68'].values[0])
						fig.axes[iax].errorbar(xgs[ss],point-mubar*subtract_mean,yerr=[[e68*bunchindex],[e68*(1-bunchindex)]],     color=colours[igal],marker=markers[igal],markersize=markersize,alpha=alpha,elinewidth=elw,markeredgewidth=mew,          linestyle='none',capsize=capsize, label=lab)
						#e95 = float(dfsummaries[dfsummaries['SN']==SNe[ss]][f'{PAR}95'].values[0])
						#fig.axes[0].errorbar(xgs[ss],point-mubar*subtract_mean,yerr=[[e95*bunchindex],[e95*(1-bunchindex)]],     color=colours[igal],marker=markers[igal],markersize=0,alpha=alpha,elinewidth=elw,markeredgewidth=mew          		  linestyle='none',capsize=capsize*0.5, elinewidth=0.25)
					else:
						if g_or_z=='z' and PAR=='mu':#Overlay sigmaext
							fig.axes[iax].errorbar(xgs[ss],mu-mubar*subtract_mean,yerr=mu_full_errs[ss],color=colours[igal],marker=None,markersize=0,alpha=0.5,elinewidth=elw,markeredgewidth=mew,linestyle='none',capsize=capsize)
							#Plot zcmb
							zcmb = self.dfmus[self.dfmus['SN']==SNe[ss]]['zcmb_hats'].values[0]
							mucmb = self.dfmus[self.dfmus['SN']==SNe[ss]]['muext_zcmb_hats'].values[0]
							fig.axes[iax].errorbar(xgs[ss]-x_coords[igal]+zcmb,mu-mucmb,yerr=err,color=colours[igal],marker=markers[igal],markersize=markersize,alpha=0.2,elinewidth=elw,markeredgewidth=mew,linestyle='none',capsize=capsize)
							mu_zcmb_full_errs = np.array([get_muext_err(zhelerr,zcmb)**2+muerr**2 for muerr,zhelerr in zip(muerrs,dfgal['zhelio_errs'].values)])**0.5
							fig.axes[iax].errorbar(xgs[ss]-x_coords[igal]+zcmb,mu-mucmb,yerr=mu_zcmb_full_errs[ss],color=colours[igal],marker=None,markersize=0,alpha=0.1,elinewidth=elw,markeredgewidth=mew,linestyle='none',capsize=capsize)

						fig.axes[iax].errorbar(xgs[ss],mu-mubar*subtract_mean,yerr=err,     										color=colours[igal],marker=markers[igal],markersize=markersize,alpha=alpha,elinewidth=elw,markeredgewidth=mew,          linestyle='none',capsize=capsize, label=lab)


					#Potential to add in distance errors with total intrinsic scatter
					if plot_full_errors and PAR=='mu':	fig.axes[0].errorbar(xgs[ss],mu-mubar,yerr=fullerrors[ss], color=colours[igal],marker=markers[igal],markersize=markersize,alpha=0.4,linestyle='none',capsize=capsize)

					if annotate_mode=='text':
						dd,ha = self.get_dxgs(Sg,ss,g_or_z)#For SN labelling
						fig.axes[iax].annotate(SNe[ss][text_index:],xy=(xgs[ss]+dd,mu-mubar),weight='bold',fontsize=self.FS-4,ha=ha)

			if PAR=='AV' and not subtract_mean:
				fig.axes[iax].set_ylim([0,None])
			if PAR=='theta' and not subtract_mean:
				fig.axes[iax].plot([-1,len(Ng)],[-1.5,-1.5],c='black',linewidth=1,linestyle='--')
				fig.axes[iax].plot([-1,len(Ng)],[2,2],c='black',linewidth=1,linestyle='--')
			if PAR=='mu':
				ymax = np.amax(np.abs(fig.axes[iax].get_ylim()))
				fig.axes[iax].set_ylim([-ymax,ymax])
			if g_or_z=='g':
				fig.axes[iax].set_ylabel(ylabel,fontsize=self.FS)
				fig.axes[iax].plot([-1,len(Ng)],[0,0],c='black',linewidth=1)
				fig.axes[iax].set_xticklabels("")
			else:
				fig.axes[iax].plot([0,self.dfmus[zword].max()+0.01],[0,0],c='black',linewidth=1)
				fig.axes[iax].set_ylabel(ylabel,fontsize=self.FS)
				if PAR=='mu':
					fig.axes[iax].set_ylabel('Hubble Residuals (mag)',fontsize=self.FS)
					zHDs = np.linspace(0,self.dfmus[zword].max()+0.01,1000)
					sigmu = (5/(zHDs*np.log(10)))*(plot_sigmapec*1e3/self.c_light)
					#ymax = np.amax(np.abs(fig.axes[iax].get_ylim()))
					#fig.axes[iax].set_ylim([-ymax,ymax])
					fig.axes[iax].plot(zHDs, sigmu,linestyle='--',color='black')
					fig.axes[iax].plot(zHDs,-sigmu,linestyle='--',color='black')

			fig.axes[iax].tick_params(labelsize=self.FS-2)

		#Plot aesthetics depend on g_or_z mode
		if g_or_z=='g':
			fig.axes[-1].set_xticks(x_coords)
			fig.axes[-1].set_xticklabels(Ng)
			fig.axes[-1].set_xlabel("GalaxyID",fontsize=self.FS)
			fig.axes[-1].set_xlim([-1,len(Ng)])
		if g_or_z=='z':
			zworddict = dict(zip(['zHD_hats','zcmb_hats'],[r"$z_{\rm{HD}}$",r"$z_{\rm{CMB}}$"]))
			fig.axes[-1].set_xlabel(zworddict[zword],fontsize=self.FS)
			fig.axes[-1].set_xlim([0,self.dfmus[zword].max()+0.01])

		if annotate_mode=='legend':
			#If/else because latter is weird fig rescaling for single panel plot
			if not multiplot:
				#if g_or_z=='z':#Remove errorbars from legend
				lines, labels = fig.axes[-1].get_legend_handles_labels()
				lines = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in lines]
				fig.axes[0].legend(lines,labels,fontsize=self.FS-2,title='SN Siblings', **args_legend,title_fontsize=self.FS)
				#else:
				#	fig.axes[0].legend(fontsize=self.FS-2,title='SN Siblings', **args_legend,title_fontsize=self.FS)
			else:
				lines, labels = fig.axes[-1].get_legend_handles_labels()
				if g_or_z=='z':#Remove errorbars from legend
					lines = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in lines]
				fig.legend(lines, labels, fontsize=self.FS-2,title='SN Siblings', **args_legend,title_fontsize=self.FS)

		#if multiplot:
		self.FS += -len(PARS)*FAC
		#fig.subplots_adjust(top=0.9)
		#fig.subplots_adjust(wspace=0, hspace=0)
		pl.tight_layout()
		if save:
			pl.savefig(f"{self.plotpath}{PAR if not multiplot else 'Parameter'}s_{g_or_z if g_or_z!='z' else zword}.pdf", bbox_inches="tight")
		if show:
			pl.show()


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

	def __init__(self,mus,errors,names,galname,prior_upper_bounds=[0.1,0.15,1.0],sigma0=0.094,sigR_res=0.00025,rootpath='./',fontsize=18,show=False,save=True):
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

		#Ngrid : int (optional; default=1000)
		#	the number of sigmaRel grid points used in prior for computing posterior in range[0,prior_upper_bound]
		sigR_res : float (optional; default=0.00025)
			the resolution in sigmaRel grid used in prior for computing posterior in range[0,prior_upper_bound]

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
		#self.Ngrid   = Ngrid
		self.sigR_res = sigR_res
		self.FS       = fontsize
		self.save     = save
		self.show     = show

		self.Sg         = int(len(mus))
		self.fullerrors = np.array([self.sigma0**2 + err**2 for err in self.errors])**0.5

		self.n_warmup   = 1000
		self.n_sampling = 25000
		self.n_chains   = 4

		#Paths
		self.packagepath  = os.path.dirname(os.path.abspath(__file__))#The package path
		self.rootpath     = rootpath  #The project rootpath
		self.modelpath    = self.rootpath  + 'model_files/' #Where model_files will be stored (copied to from package)
		self.stanpath     = self.modelpath + 'stan_files/'
		self.productpath  = self.rootpath  + 'products/'
		self.plotpath     = self.rootpath  + 'plots/single_galaxy_plots/'
		self.create_paths()
		try:
			shutil.copytree(os.path.join(self.packagepath,'stan_files'), self.modelpath+'stan_files')#Copy stan_files from packagepath to modelpath (if local dev. these are the same)
		except:
			print (f"Tried copying stan_files folder from :{os.path.join(self.packagepath,'stan_files')} to {self.modelpath+'stan_files'}")
			print ("But the latter folder already exists.")

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
			den = (2*np.pi*(sig**2))**0.5
			return num/den

		#Check prior distribution is correct
		if prior_distribution!='uniform': raise Exception('Not yet implemented other prior distributions')

		#Get prior grid and prior normalisation
		#self.sigRs      = np.linspace(0, prior_upper_bound, self.Ngrid)
		self.sigRs      = np.arange(0,prior_upper_bound+self.sigR_res,self.sigR_res)
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

	def combine_individual_distances(self,mode=None,overwrite=True,asymmetric=False):
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

		asymmetric : bool or None (optional; default=False)
			choice of sigmaRel hyperprior, False use Arcsine on rho, True use sigR~U(0,sig0)
			if None, set to attribute if there or False otherwise

		End Product(s)
		----------
		self.common_distance_estimates : dict
			{key,value} are the mode and x
			where x is median and std of common distance estimate in dict form

		self.STORE : dict
			keys are modes, values.keys() are ['data','summary','chains'] from posterior fits

		"""
		#SigmaRel Hyperprior for dM-Mixed fit
		if asymmetric is None and 'asymmetric' not in self.__dict__.keys():
			self.asymmetric = False
		elif asymmetric in [False,True]:
			self.asymmetric = asymmetric

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
		if not os.path.exists(self.productpath+f"FITS{self.galname}{'_asym' if self.asymmetric else ''}.pkl") or overwrite:
			#For each mode, perform stan fit to combine distances
			STORE = {} ; FITS = {}
			print ('###'*30)
			for mode in modes:
				print (f"Beginning Stan fit adopting the dM-{mode.capitalize()} assumption")
				stan_data = dict(zip(['S','sigma0','mean_mu','mu_s','mu_err_s','asymmetric'],[self.Sg,self.sigma0,np.average(self.mus),self.mus,self.errors,int(self.asymmetric)]))
				model = CmdStanModel(stan_file=self.stanpath+stan_files[mode])
				fit         = model.sample(data=stan_data,chains=self.n_chains, iter_sampling=self.n_sampling, iter_warmup = self.n_warmup, seed=42)
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
			with open(self.productpath+f"FITS{self.galname}{'_asym' if self.asymmetric else ''}.pkl",'wb') as f:
				pickle.dump({'FITS':FITS,'common_distance_estimates':STORE},f)
		else:#Else load up
			with open(self.productpath+f"FITS{self.galname}{'_asym' if self.asymmetric else ''}.pkl",'rb') as f:
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

	def plot_sigmaRel_posteriors(self,xupperlim=0.25,colours=None,blind=False,fig_ax=None):
		"""
		Plot Sigma_Rel Posteriors

		Method to overlay each sigmaRel posterior from each choice of prior upper bound

		Parameters
		----------
		xupperlim : float or 'adaptive' (optional; default=0.25)
			define maximum x-value (i.e. sigmaRel value) on plot for visualisation purposes only

		colours : lst (optional; default=None)
			colours for each sigmaRel overlay, defaults to ['green','purple','goldenrod']

		blind : bool (optional; default=False)
			if True, blind sigmaRel plot axes and posterior summaries

		fig_ax : None or list (optional; default=None)
			if None, create new figure, else, fig_ax = [fig,ax,counter], where counter denotes how many times figure has been used

		Returns
		----------
		plot of sigmaRel posterior overlays
		"""
		alph = 0.2 ; dfs = 3
		if 'posteriors' not in list(self.__dict__.keys()):
			self.get_sigmaRel_posteriors()
		if colours is None:
			colours    = ['green','purple','goldenrod']
			linestyles = ['-','--',':','-.']

		if fig_ax is None:#Create new figure
			fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
			pl.title(r'$\sigma_{\rm{Rel}}$-Posterior from Individual Distance Estimates',fontsize=self.FS+1, weight='bold')
			counter = 0 ; kmax = 0 ; leglabel = ''
		else:#Load old figure
			try:
				fig,ax,counter,leglabel,kmax = fig_ax[:]
			except:
				fig,ax,counter,leglabel = fig_ax[:]
				kmax = 0


		XQs = {0.005:[],0.995:[]}
		if kmax!=0:
			kfac = max([np.amax(posterior) for posterior in self.posteriors.values()])
			self.posteriors = {key:value*kmax/kfac for key,value in self.posteriors.items()}

		for ip,prior_upper_bound in enumerate(self.prior_upper_bounds):
			ip += counter
			try:	ccc = colours[ip]; lw = linestyles[ip] if fig_ax is not None else '-'
			except:	ccc = f'C{ip%10}'; lw = linestyles[ip%len(linestyles)] if fig_ax is not None else '-'
			self.posterior = self.posteriors[prior_upper_bound]
			self.sigRs     = self.sigRs_store[prior_upper_bound]
			#Plots
			fig.axes[0].plot(self.sigRs, self.posterior,c=ccc,label=leglabel,linestyle=lw)
			fig.axes[0].plot([self.sigRs[-1],self.sigRs[-1]],[0,self.posterior[-1]],linestyle='--',c=ccc)
			kmax = max([kmax,np.amax(self.posterior)])
			#Begin Annotations
			Xoff = 0.65 ; dX   = 0.08 ; Yoff = 0.845 ; dY   = 0.07 ; dyoff = -0.275001
			#~~~
			#if leglabel=='':
			if leglabel!='':
				Yoff += -0.2
			fig.axes[0].annotate('Hyperprior',xy=(Xoff,Yoff+dY),			   xycoords='axes fraction',fontsize=15.5, ha='left')
			LABEL = r'$\sigma_{\rm{Rel}} \sim U (0,%s)$'%(str(round(float(prior_upper_bound),3)))
			fig.axes[0].annotate(LABEL,xy=(Xoff,Yoff-dY*ip),xycoords='axes fraction',fontsize=self.FS-dfs,color=ccc,ha='left')
			#~~~
			fig.axes[0].annotate('Posterior Summary', xy=(Xoff,Yoff+dY-0.275001*1.125),xycoords='axes fraction',fontsize=15.5, ha='left')
			#Decide how to summarise posterior
			KDE = copy.deepcopy(self.posterior)
			imode = np.argmax(KDE)
			xmode = self.sigRs[imode] ; KDEmode = KDE[imode]
			condition1 = np.argmax(KDE)!=0 and np.argmax(KDE)!=len(KDE)-1#KDE doesnt peak at prior boundary
			hh = np.exp(-1/8)#Gaussian height at sigma/2 #KDE is not near flat topped at prior boundary
			condition2 = not (KDE[0]>=hh*KDEmode or KDE[-1]>=hh*KDEmode)

			Yoff += dyoff*1.125
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
				if blind: summary = ['0.X','0.X','0.X']
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
				if blind:	s68,s95 = '0.X','(0.X)'
				fig.axes[0].annotate("%s %s %s %s"%(r'$\sigma_{\rm{Rel}}$',lg, s68, s95),xy=(Xoff,Yoff-dY*ip),xycoords='axes fraction',fontsize=self.FS-dfs,color=ccc,ha='left')
			Yoff +=  dyoff

		fig.axes[0].set_yticks([])
		if xupperlim!='adaptive':
			fig.axes[0].set_xlim([0,xupperlim])
			fig.axes[0].set_xticks(np.arange(0,0.25,0.05))
		else:
			qmin,qmax = min(list(XQs.keys())),max(list(XQs.keys()))
			DX = max(XQs[qmax]) - min(XQs[qmin]) ; fac = 0.1
			fig.axes[0].set_xlim([max([0,min(XQs[qmin])-DX*fac]),max(XQs[qmax])+DX*2*fac])
		if blind:	fig.axes[0].set_xticks([])
		fig.axes[0].set_ylabel(r'Posterior Density',fontsize=self.FS)
		fig.axes[0].set_xlabel(r'$\sigma_{\rm{Rel}}$ (mag)',fontsize=self.FS)#fig.text(0, 0.5, 'X', rotation=90, va='center', ha='center',color='white',fontsize=100)#fig.text(-0.06, 0.5, 'Posterior Density', rotation=90, va='center', ha='center',color='black',fontsize=self.FS)
		pl.tick_params(labelsize=self.FS)
		YMIN,YMAX = list(fig.axes[0].get_ylim())[:]
		fig.axes[0].set_ylim([0,YMAX])
		if self.save:
			if leglabel!='':	pl.legend(loc='upper right',fontsize=self.FS-2,framealpha=1)
			pl.tight_layout()
			pl.savefig(f"{self.plotpath}{self.galname}_SigmaRelPosteriors.pdf",bbox_inches="tight")
		if self.show:
			pl.show()
		if fig_ax is not None:
			self.fig_ax = [fig,ax,kmax]
		else:
			self.fig_ax = fig_ax

	def plot_common_distances(self,markersize=10,capsize=8,mini_d=0.025,asymmetric=None):
		"""
		Plot Common Distances

		Parameters
		----------
		markersize : float (optional; default=10)
			size of distance scatter point markers

		capsize : float (optional; default=8)
			size of errorbar caps

		mini_d : float (optional; default=0.025)
			delta_x value for separating distances visually

		asymmetric : bool or None (optional; default=None)
			choice of sigmaRel hyperprior, False use Arcsine on rho, True use sigR~U(0,sig0)
			if None, set to attribute if there or False otherwise
		"""
		if asymmetric is None and 'asymmetric' not in self.__dict__.keys():
			self.asymmetric = False
		elif asymmetric in [False,True]:
			self.asymmetric = asymmetric

		fig,ax = pl.subplots(1,1,figsize=(8,6),sharex='col',sharey=False,squeeze=False)
		fig.axes[0].set_title(r"Combination of Individual Distance Estimates",weight='bold',fontsize=self.FS+0.5)

		Nmodes = int(len(self.common_distance_estimates))
		deltas = [mini_d*(nn-(Nmodes-1)/2) for nn in range(Nmodes)]
		markers = [m for m in ['o','^','s'][:Nmodes]]
		colors  = ['r','g','b']
		mode_labels = [f"$\delta M$-{mode.capitalize()}" for mode in self.common_distance_estimates]
		sig_labels  = [r'$\sigma_{\rm{Rel}}=\sigma_0$',
					   r'$\rho \sim \rm{Arcsine}(0,1)$',
					   r'$\sigma_{\rm{Rel}}=0$']
		if self.asymmetric: sig_labels[1] = r'$\sigma_{\rm{Rel}}\sim U(0,\sigma_0)$'

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
		fig.axes[0].annotate(r'$\sigma_{\rm{Rel}}$-Hyperpriors:',
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
			pl.savefig(f"{self.plotpath}{self.galname}_CommonDistances{'_asym' if self.asymmetric else ''}.pdf", bbox_inches="tight")
		if self.show:
			pl.show()
