"""
ModelLoader Class

Module containing ModelLoader class
Methods are useful for preparing data and model before posterior computation


Contains:
--------------------
ModelLoader class:
	inputs: sigma0, sigmapec, eta_sigmaRel_input, use_external_distances, zmarg, alt_prior, zcosmo, alpha_zhel, choices

	Methods are:
		get_model_params()
		get_modelkey()
		update_stan_file()
		get_stan_data(dfmus)
		get_stan_init(dfmus, savepath, n_chains)
		get_pars()
--------------------

Written by Sam M. Ward: smw92@cam.ac.uk
"""

import copy, json, re
import numpy as np

class ModelLoader:

	def __init__(self, sigma0, sigmapec, eta_sigmaRel_input, use_external_distances, zmarg, alt_prior, zcosmo, alpha_zhel, choices):
		"""
		Initialisation

		Parameters
		----------
		sigma0 : float or None
			the total intrinsic scatter, i.e. informative prior upper bound on sigmaRel

		sigmapec : float or None
			same as for sigma0, default value is 250 km/s

		eta_sigmaRel_input : float or None
			option to fix sigmaRel at a fraction of sigma0
			if None, free sigmaRel
			if float, assert between 0 and 1

		use_external_distances : bool or None
			option to include external distance constraints

		zmarg : bool
			if False, use LCDM distances and Gaussian uncertainty approximation in distance
			if True, use z_g data and draw Gaussian z_g parameters

		alt_prior : bool
			two default choices of prior are [sigma0~Uniform and rho~Arcsine(0,1)] OR [sigmaRel~Prior; sigmaCommon~Prior]
			alt_prior=False is former, alt_prior=True is latter
			Other choices include:
				-A: sigR ~ U(0,sig0)
				-B: sigC ~ U(0,sig0)
				-C: rho  ~ U(0,1)

		zcosmo : str
			choice of zHD or zcmb

		alpha_zhel : bool (optional; default=False)
			if zmarg is True, then alpha_zhel can be activated. This takes the pre-computed slopes of dmu_phot = alpha*dzhelio and marginalises over this in the z-pipeline

		choices : class
			class with attributes with pre-defined values (user input or None)
		"""
		self.sigma0                 = sigma0
		self.sigmapec               = sigmapec
		self.eta_sigmaRel_input     = eta_sigmaRel_input
		self.use_external_distances = use_external_distances
		self.zmarg					= zmarg
		self.alt_prior              = alt_prior
		if self.alt_prior=='arcsine': self.alt_prior=False#Simple mapping, arcsine hyperprior on rho
		if self.alt_prior=='uniform': self.alt_prior='C'  #Simple mapping, uniform hyperprior on rho
		self.zcosmo                 = zcosmo
		self.alpha_zhel             = alpha_zhel

		self.choices  = copy.deepcopy(choices) #Choices inputted in class call
		self.stanpath = self.choices.stanpath

	def get_model_params(self):
		"""
		Get Model Parameters

		Takes model choices and sets class attributes appropriately

		End Product(s)
		----------
		self.sigma0
		self.sigmapec
		self.sigmaRel_input
		self.eta_sigmaRel_input
		self.use_external_distances
		"""

		#Get values for fit
		if self.sigma0 is None:
			self.sigma0   = self.choices.sigma0	 #If not specified, use the class input value (which if itself is not specified has a default value sigma0=0.1)
		if self.sigmapec is None:
			self.sigmapec = self.choices.sigmapec #If not specified, use the class input value (which if itself is not specified has a default value sigmapec=250)
		if self.eta_sigmaRel_input is None:
			self.eta_sigmaRel_input = self.choices.eta_sigmaRel_input
		if self.eta_sigmaRel_input is None:
			self.sigmaRel_input     = 0  #If not specified, free sigmaRel
		else:#Otherwise, fix sigmaRel to fraction of sigma0
			assert(type(self.eta_sigmaRel_input) in [float,int])
			assert(0<=self.eta_sigmaRel_input and self.eta_sigmaRel_input<=1)
			self.sigmaRel_input     = 1
		if self.use_external_distances is None:
			self.use_external_distances = self.choices.use_external_distances
		if self.zcosmo is None:
			self.zcosmo = self.choices.zcosmo

	def get_modelkey(self):
		"""
		Get Model Key

		Takes input choices and gets appropriate data, builds a filename called modelkey

		End Product(s)
		----------
		Prints out model choices

		self.modelkey : str
			descriptive filename of model choice
		"""

		#Get Stan HBM files for the different intrinsic scatter modelling assumptions
		sigmas   = {'sigma0'	:	{'value':self.sigma0},
					'sigmapec'	:	{'value':self.sigmapec}
					}
		print ('###'*10)
		for label in sigmas.keys():
			value = sigmas[label]['value']
			if type(value) in [float,int] and value>0:
				sigmastr = f"fixed_{label}{value}"
				if (self.use_external_distances and label=='sigmapec') or label in ['sigma0']:
					print (f"{label} fixed at {value}")
			elif value=='free':
				sigmastr = f"free_{label}"
				if (not self.use_external_distances and label=='sigmapec'):
					raise Exception(f"Freeing {label} without external distances")
				print (f"{label} is a free hyperparameter")
			else:
				raise Exception(f"{label} must be float/int >0 or 'free', but {label}={value}")
			sigmas[label]['str'] = sigmastr
		muextstr = {False:'no_muext',True:f'with_muext{self.zcosmo}'}[self.use_external_distances]

		#Model being used
		modelkey = f"{sigmas['sigma0']['str']}_{muextstr}"
		if sigmas['sigma0']['value']!='free' and not self.use_external_distances:
			print('No external distances used')
		elif sigmas['sigma0']['value']=='free' and not self.use_external_distances:
			raise Exception('Freeing sigma0 without external distances')
		else:
			print ('Using external distances')
			modelkey += f"_{sigmas['sigmapec']['str']}"
		if bool(self.sigmaRel_input):
			print (f"sigmaRel is fixed at {self.eta_sigmaRel_input}*sigma0")
			modelkey += f"_etasigmaRelfixed{self.eta_sigmaRel_input}"
		else:
			print (f"sigmaRel is free hyperparameter")
			modelkey += f"_sigmaRelfree"
		if self.zmarg:
			print ("Marginalising over z_g parameters")
			modelkey += f"_zmarg"
			if self.alpha_zhel:
				print ("Marginalising over large zhelio error galaxies")
				modelkey += f"_alphazhelio"
		if self.alt_prior:
			print (f"Using Alternative Prior"+str(self.alt_prior)*(self.alt_prior!=True))
			if self.alt_prior=='p': print (f'p={self.choices.p}; usamp_input={self.choices.usamp_input}')
			modelkey += f"_altprior"
			if self.alt_prior!=True:
				modelkey+= f'{self.alt_prior}'

		self.modelkey = modelkey
		print ('###'*10)

		"""#Description here of potential modelkeys being made, and their base .stan files
		 POTENTIAL stan_files		:	 MODELKEYS
		'sigmaRel_nomuext.stan'		:	'fixed_sigma0_no_muext'
		'sigmaRel_withmuext.stan'	:	'fixed_sigma0_with_muext_fixed_sigmapec'
										'fixed_sigma0_with_muext_free_sigmapec'
										'free_sigma0_with_muext_fixed_sigmapec'
										'free_sigma0_with_muext_free_sigmapec'
		For each we can have sigmaRelfree OR etasigmaRelfixed
		"""

	def update_stan_file(self):
		"""
		Update stan file

		Method to edit .stan file according to model fit choices

		End Product(s)
		----------
		Saves current_model.stan file with appropriately edited lines
		"""
		#Update stan file according to choices, create temporary 'current_model.stan'
		if self.alt_prior in [False,True]:
			self.stan_filename = {True:{False:'sigmaRel_withmuext.stan',True:'sigmaRel_withmuext_alt.stan'}[self.alt_prior],False:'sigmaRel_nomuext.stan'}[self.use_external_distances]
		else:
			self.stan_filename = f'altpriors/sigmaRel_withmuext_alt{self.alt_prior}.stan'
		#self.stan_filename = {True:{False:'sigmaRel_withmuext.stan',True:'sigmaRel_withmuext_alt5.stan'}[self.alt_prior],False:'sigmaRel_nomuext.stan'}[self.use_external_distances]
		if self.zmarg:	self.stan_filename = self.stan_filename.replace('.stan','_zmarg.stan')
		with open(self.stanpath+self.stan_filename,'r') as f:
			stan_file = f.read().splitlines()
		if ('fixed_sigmapec' in self.modelkey or 'fixed_sigma0' in self.modelkey) and self.use_external_distances:
			for il, line in enumerate(stan_file):
				if 'fixed_sigmapec' in self.modelkey:
					if bool(re.match(r'\s*//real<lower=0,upper=1>\s*pec_unity;\s*//Data',line)):
						stan_file[il] = line.replace('//real','real')
					if bool(re.match(r'\s*real<lower=0,upper=1>\s*pec_unity;\s*//Model',line)):
						stan_file[il] = line.replace('real','//real')
					if bool(re.match(r'\s*pec_unity\s*~\s*uniform\(0,1\)',line)):
						stan_file[il] = line.replace('pec_unity','//pec_unity')
				if 'fixed_sigma0' in self.modelkey:
					if bool(re.match(r'\s*//real<lower=0,upper=1>\s*sigma0;\s*//Data',line)):
						stan_file[il] = line.replace('//real','real')
					if bool(re.match(r'\s*real<lower=0,upper=1>\s*sigma0;\s*//Model',line)):
						stan_file[il] = line.replace('real','//real')
					if bool(re.match(r'\s*sigma0\s*~\s*uniform\(0,1\)',line)):
						stan_file[il] = line.replace('sigma0','//sigma0')
		stan_file = '\n'.join(stan_file)#The model required for current fit
		create_new_model = True
		try:
			with open(self.stanpath+'current_model.stan','r') as f:#The .stan file corresponding to current_model which is already compiled
				current_stan_file = f.read()
			if current_stan_file==stan_file:#If new model is same as current_model.stan, do nothing
				create_new_model = False
		except:#Likely no current_model.stan exists
			pass
		if create_new_model:#Else, create new current_model.stan
			print (stan_file)
			with open(self.stanpath+'current_model.stan','w') as f:
				f.write(stan_file)

	def get_stan_data(self, dfmus):
		"""
		Get Stan Data

		Uses dfmus dataframe to prepare stan_data dictionary

		Parameters
		----------
		dfmus : pandas Dataframe
			data, columns are ['Galaxy','SN','mus','mu_errs','muext_hats','zcosmo_hats','zcosmo_errs']

		Returns
		----------
		stan_data : dict
		 	dictionary with data to fit
		"""
		#Cosmo for marginalising over z and computing luminosity distances in model
		#H0 = 73.24 ; Om0 = 0.28 ; Ol0 = 0.72 ; c_light = 299792458
		c_light = self.choices.c_light ; H0  = self.choices.fiducial_cosmology['H0'] ; Om0 = self.choices.fiducial_cosmology['Om0']
		Ol0 = 1-Om0
		q0  = Om0/2 - Ol0 ; j0 = Om0 + Ol0 ; c_H0 = c_light/(H0*1e3)

		print ('###'*30)
		print (f"Beginning Stan fit: {self.modelkey}")
		#Groupby galaxy to count Nsibs per galaxy
		Gal   = dfmus.groupby('Galaxy',sort=False)[[f'muext_{self.zcosmo}_hats',f'{self.zcosmo}_hats',f'{self.zcosmo}_errs']].agg('mean')
		Ng    = Gal.shape[0]
		S_g   = dfmus.groupby('Galaxy',sort=False)['SN'].agg('count').values
		Nsibs = int(sum(S_g))
		for sg in S_g:
			assert(sg>=2)
		#Individual siblings distance estimates
		mu_sib_phots     = dfmus['mus'].values
		mu_sib_phot_errs = dfmus['mu_errs'].values
		#External Distances
		mu_ext_gal, zcosmos, zcosmoerrs = [Gal[col].tolist() for col in Gal.columns]

		def etamapper(x): return 0 if x is None else x
		#Load up data
		stan_data = dict(zip(['Ng','S_g','Nsibs','mu_sib_phots','mu_sib_phot_errs','mu_ext_gal','zcosmos','zcosmoerrs','sigmaRel_input','eta_sigmaRel_input'],
							 [ Ng , S_g , Nsibs , mu_sib_phots , mu_sib_phot_errs , mu_ext_gal , zcosmos , zcosmoerrs , self.sigmaRel_input , etamapper(self.eta_sigmaRel_input) ]))
		if not self.use_external_distances or self.zmarg:
			stan_data = {key:value for key,value in stan_data.items() if key not in ['mu_ext_gal','zcosmos','zcosmoerrs']}

		if self.zmarg:
			zhelios = dfmus.groupby('Galaxy',sort=False)['zhelio_hats'].agg('mean').tolist()
			stan_data['zhelio_hats'] = zhelios
			stan_data['zpo_hats']    = np.asarray(zhelios)-np.asarray(zcosmos)
			stan_data['zhelio_errs'] = zcosmoerrs#Assume measurements errors on zcosmo and zhelio are the same, and they are perfectly correlated
			stan_data['q0'] = q0 ; stan_data['j0'] = j0 ; stan_data['c_H0'] = c_H0

			if self.alpha_zhel:
				zhel_sibs = dfmus[dfmus['alpha_mu_z']!=0].index
				zhel_sibs_indices = [_ for _,sn in enumerate(dfmus.index) if sn in zhel_sibs]
				stan_data['Nzhel']      = int(len(zhel_sibs))
				stan_data['Nzhelgal']   = dfmus.loc[zhel_sibs]['Galaxy'].nunique()#Purely for plotting purposes to describe how many galaxies were modelled this way
				stan_data['alpha_zhel'] = dfmus.loc[zhel_sibs]['alpha_mu_z'].to_list()
				Qsz = np.zeros((stan_data['Nsibs'],stan_data['Nzhel']))
				for _ in range(dfmus.shape[0]):
					if _ in zhel_sibs_indices:
						Qsz[_,zhel_sibs_indices.index(_)] = 1
				stan_data['Q_sib_zhel'] = Qsz
			else:
				stan_data['Nzhel'] = 0 ; stan_data['Nzhelgal'] = 0; stan_data['alpha_zhel'] = [] ; stan_data['Q_sib_zhel'] = np.zeros((stan_data['Nsibs'],0))

		if self.sigma0!='free':
			stan_data['sigma0']    = self.sigma0
		if self.sigmapec!='free':
			stan_data['pec_unity'] = self.sigmapec*1e3/self.choices.c_light
		if self.alt_prior=='p':
			stan_data['p']           = self.choices.p
			stan_data['usamp_input'] = self.choices.usamp_input

		return stan_data

	def get_stan_init(self, dfmus, savepath, n_chains):
		"""
		Get Stan Initialisations

		Important for fits with redshift parameters, often gets stuck if z not initialised
		Cmdstanpy works by loading in a list of filenames, one for each chain. Each filename is a json corresponding to a dictionary with values that are the initialisations

		Parameters
		----------
		dfmus : pandas Dataframe
			data, columns are ['Galaxy','zcmb_hats']

		savepath : str
			path where json files are saved/loaded

		n_chains : int
			no. of chains

		Returns
		----------
		stan_init : list of str
			jsons files with initialisations
		"""
		if self.zmarg:#zh_guesses = dfmus.groupby('Galaxy')['zhelio_hats'].agg('mean').to_numpy()#zg_param_guesses = np.vstack((zh_guesses,zc_guesses)).T#zg_param_guesses = dfmus.groupby('Galaxy')[['zhelio_hats','zcmb_hats']].agg('mean').to_numpy().tolist()#json_file = {'zg_param':zg_param_guesses,'pec_unity':250/3e5}#,'nuhelio':np.ones(stan_data['Ng']).tolist()}
			zc_guesses = dfmus.groupby('Galaxy')[f'{self.zcosmo}_hats'].agg('mean').to_numpy().tolist()
			json_file = {'zcosmos':zc_guesses,'pec_unity':250/3e5}
		else:
			json_file = {}

		with open(f"{savepath}inits.json", "w") as f:
			json.dump(json_file, f)

		stan_init  = [f"{savepath}inits.json" for _ in range(n_chains)]
		return stan_init

	def get_pars(self):
		"""
		Get parameters

		Based on information, drop parameters if they are in fact fixed

		Returns
		----------
		pars with appropriate parameter names removed
		"""
		pars=['rho','sigma0','sigmaRel','sigmaCommon','sigmapec']
		if self.sigma0!='free':
			pars.remove('sigma0')
			pars.remove('rho')
			pars.remove('sigmaCommon')
		if self.sigmapec!='free':
			pars.remove('sigmapec')
		return pars
