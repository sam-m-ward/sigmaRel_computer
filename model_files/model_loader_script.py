"""
ModelLoader Class

Module containing ModelLoader class
Methods are useful for preparing data and model before posterior computation


Contains:
--------------------
ModelLoader class:
	inputs: sigma0, sigmapec, eta_sigmaRel_input, use_external_distances, zmarg, alt_prior, choices

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

	def __init__(self, sigma0, sigmapec, eta_sigmaRel_input, use_external_distances, zmarg, alt_prior, choices):
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
			two choices of prior are [sigma0~Prior and sigmaRel~U(0,sigma0)] OR [sigmaRel~Prior; sigmaCommon~Prior]
			alt_prior=False is former, alt_prior=True is latter

		choices : class
			class with attributes with pre-defined values (user input or None)
		"""
		self.sigma0                 = sigma0
		self.sigmapec               = sigmapec
		self.eta_sigmaRel_input     = eta_sigmaRel_input
		self.use_external_distances = use_external_distances
		self.zmarg					= zmarg
		self.alt_prior              = alt_prior

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
		muextstr = {False:'no_muext',True:'with_muext'}[self.use_external_distances]

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
		if self.alt_prior:
			print (f"Using Alternative Prior")
			modelkey += f"_altprior"
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
		self.stan_filename = {True:{False:'sigmaRel_withmuext.stan',True:'sigmaRel_withmuext_alt.stan'}[self.alt_prior],False:'sigmaRel_nomuext.stan'}[self.use_external_distances]
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
		stan_file = '\n'.join(stan_file)
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
			data, columns are ['Galaxy','SN','mus','mu_errs','muext_hats','zcmb_hats','zcmb_errs']

		Returns
		----------
		stan_data : dict
		 	dictionary with data to fit
		"""
		#Cosmo for marginalising over z and computing luminosity distances in model
		H0 = 73.24 ; Om0 = 0.28 ; Ol0 = 0.72 ; c_light = 299792458
		q0 = Om0/2 - Ol0 ; j0 = Om0 + Ol0 ; c_H0 = c_light/(H0*1e3)

		print ('###'*30)
		print (f"Beginning Stan fit: {self.modelkey}")
		#Groupby galaxy to count Nsibs per galaxy
		Gal   = dfmus.groupby('Galaxy',sort=False)[['muext_hats','zcmb_hats','zcmb_errs','zhelio_hats']].agg('mean')
		Ng    = Gal.shape[0]
		S_g   = dfmus.groupby('Galaxy',sort=False)['SN'].agg('count').values
		Nsibs = int(sum(S_g))
		for sg in S_g:
			assert(sg>=2)
		#Individual siblings distance estimates
		mu_sib_phots     = dfmus['mus'].values
		mu_sib_phot_errs = dfmus['mu_errs'].values
		#External Distances
		mu_ext_gal, zcmbs, zcmberrs, zhelios = [Gal[col].tolist() for col in Gal.columns]

		def etamapper(x): return 0 if x is None else x
		#Load up data
		stan_data = dict(zip(['Ng','S_g','Nsibs','mu_sib_phots','mu_sib_phot_errs','mu_ext_gal','zcmbs','zcmberrs','sigmaRel_input','eta_sigmaRel_input'],
							 [ Ng , S_g , Nsibs , mu_sib_phots , mu_sib_phot_errs , mu_ext_gal , zcmbs , zcmberrs , self.sigmaRel_input , etamapper(self.eta_sigmaRel_input) ]))
		if not self.use_external_distances or self.zmarg:
			stan_data = {key:value for key,value in stan_data.items() if key not in ['mu_ext_gal','zcmbs','zcmberrs']}
		#if self.alt_prior:
		#	stan_data = {key:value for key,value in stan_data.items() if key not in ['sigmaRel_input','eta_sigmaRel_input']}
		if self.zmarg:#stan_data['zg_data']     = zcmbs#stan_data['zgerrs_data'] = zcmberrs#stan_data['zg_data']     = [np.array([zh,zc]) for zh,zc in zip(zcmbs, zhelios)]#stan_data['zgerrs_data'] = [np.ones((2,2))*ze for ze in zcmberrs]#Assume measurements errors on zcmb and zhelio are the same, and they are perfectly correlated
			stan_data['zhelio_hats'] = zhelios
			stan_data['zpo_hats']    = np.asarray(zhelios)-np.asarray(zcmbs)
			stan_data['zhelio_errs'] = zcmberrs#Assume measurements errors on zcmb and zhelio are the same, and they are perfectly correlated
			stan_data['q0'] = q0 ; stan_data['j0'] = j0 ; stan_data['c_H0'] = c_H0

		if self.sigma0!='free':
			stan_data['sigma0']    = self.sigma0
		if self.sigmapec!='free':
			stan_data['pec_unity'] = self.sigmapec*1e3/self.choices.c_light

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
			zc_guesses = dfmus.groupby('Galaxy')['zcmb_hats'].agg('mean').to_numpy().tolist()
			json_file = {'zHDs':zc_guesses,'pec_unity':250/3e5}
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
		pars=['sigma0','sigmapec','rho','sigmaRel','sigmaCommon']
		if self.sigma0!='free':
			pars.remove('sigma0')
			pars.remove('rho')
			pars.remove('sigmaCommon')
		if self.sigmapec!='free':
			pars.remove('sigmapec')
		return pars
