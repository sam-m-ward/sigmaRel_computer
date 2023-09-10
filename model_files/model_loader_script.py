"""
ModelLoader Class

Module containing ModelLoader class
Methods are useful for preparing data and model before posterior computation


Contains:
--------------------
ModelLoader class:
	inputs: sigma0, sigmapec, eta_sigmaRel_input, use_external_distances, choices

	Methods are:
		get_model_params()
		get_modelkey()
		update_stan_file()
		get_stan_data(dfmus)
		get_pars(pars=['sigmaRel','sigma0','sigmapec'])
--------------------

Written by Sam M. Ward: smw92@cam.ac.uk
"""

import copy, re

class ModelLoader:

	def __init__(self, sigma0, sigmapec, eta_sigmaRel_input, use_external_distances, choices):
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

		choices : class
			class with attributes with pre-defined values (user input or None)
		"""
		self.sigma0                 = sigma0
		self.sigmapec               = sigmapec
		self.eta_sigmaRel_input     = eta_sigmaRel_input
		self.use_external_distances = use_external_distances

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
			self.eta_sigmaRel_input = 0
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
		self.stan_filename = {True:'sigmaRel_withmuext.stan',False:'sigmaRel_nomuext.stan'}[self.use_external_distances]
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

		print ('###'*30)
		print (f"Beginning Stan fit: {self.modelkey}")
		#Groupby galaxy to count Nsibs per galaxy
		Gal   = dfmus.groupby('Galaxy',sort=False)[['muext_hats','zcmb_hats','zcmb_errs']].agg('mean')
		Ng    = Gal.shape[0]
		S_g   = dfmus.groupby('Galaxy',sort=False)['SN'].agg('count').values
		Nsibs = int(sum(S_g))
		for sg in S_g:
			assert(sg>=2)
		#Individual siblings distance estimates
		mu_sib_phots     = dfmus['mus'].values
		mu_sib_phot_errs = dfmus['mu_errs'].values
		#External Distances
		mu_ext_gal, zcmbs, zcmberrs = [Gal[col].tolist() for col in Gal.columns]

		#Load up data
		stan_data = dict(zip(['Ng','S_g','Nsibs','mu_sib_phots','mu_sib_phot_errs','mu_ext_gal','zcmbs','zcmberrs','sigmaRel_input','eta_sigmaRel_input'],
							 [ Ng , S_g , Nsibs , mu_sib_phots , mu_sib_phot_errs , mu_ext_gal , zcmbs , zcmberrs , self.sigmaRel_input , self.eta_sigmaRel_input ]))
		if not self.use_external_distances:
			stan_data = {key:value for key,value in stan_data.items() if key not in ['mu_ext_gal','zcmbs','zcmberrs']}
		if self.sigma0!='free':
			stan_data['sigma0']    = self.sigma0
		if self.sigmapec!='free':
			stan_data['pec_unity'] = self.sigmapec*1e3/self.choices.c_light

		return stan_data

	def get_pars(self,pars=['sigmaRel','sigma0','sigmapec']):
		"""
		Get parameters

		Based on information, drop parameters if they are in fact fixed

		Parameters
		----------
		pars : lst
			full list of parameter names

		Returns
		----------
		pars with appropriate parameter names removed
		"""
		if self.sigma0!='free':
			pars.remove('sigma0')
		if self.sigmapec!='free':
			pars.remove('sigmapec')
		return pars
