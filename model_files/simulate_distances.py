"""
Simulate Distances

Module contains class for simulating individual siblings distance estimates

Contains:
--------------------
SiblingsDistanceSimulator class:
	inputs: Ng=30, Sg = 2, sigma0=0.1, sigmaRel=0.05, mu_bounds = [25,35], sigma_fit_s=0.05, external_distances=False, random=None
--------------------

Written by Sam M. Ward: smw92@cam.ac.uk
"""

import numpy as np
import pandas as pd
import copy
from astropy.cosmology import FlatLambdaCDM

class SiblingsDistanceSimulator:

	def __init__(self, Ng=30, Sg = 2, sigma0=0.1, sigmaRel=0.05, zcmb_bounds = [0.01,0.1], fiducial_cosmology={"H0":73.24, "Om0":0.28}, sigma_fit_s=0.05, external_distances=False, zcmberr=1e-6, sigmapec=250, random=None):
		"""
		Initialisation

		Parameters
		----------
		Ng : int (optional; default=30)
			Number of siblings galaxies

		Sg : int or list/array of ints (optional; default=2)
			Number of siblings per galaxy

		sigma0 : float>0 (optional; default=0.1)
			total intrinsic scatter

		sigmaRel : float>0 and <sigma0 (optional; default=0.5)
			relative intrinsic scatter

		zcmb_bounds : list/array of x2 floats, each>0 (optional; default=[0.01,0.1])
			random zcmbs drawn from [zcmb_low, zcmb_high]

		fiducial_cosmology : dict (optional; default={"H0":73.24, "Om0":0.28})
			key,values are H0 and Omega_m_0

		sigma_fit_s : list/array of floats, each>0, or float (optional; default=0.05)
			the fitting uncertainties, i.e. measurement errors, on each individual siblings distance estimate

		external_distances : bool (optional; default=False)
			if True, add column of estimated external distances to pandas df

		zcmberr : float (optional; default=1e-6)
			the measurement error on redshift

		sigmapec : float (optionl; default=250)
			in km/s, the peculiar velocity dispersion

		random : None or int (optional; default=None)
			sets numpy random seed, if None use 42

		End Product(s)
		----------
		self.dfmus : pandas df
			contains the galaxy ID's and siblings, each with their individual siblings distance estimates
		"""
		self.__dict__.update(locals())
		self.sigmaCommon = np.sqrt(np.square(self.sigma0)-np.square(self.sigmaRel))
		self.rho         = np.square(self.sigmaCommon)/np.square(self.sigma0)
		self.cosmo       = FlatLambdaCDM(**fiducial_cosmology) ; c_light = 299792458/1e3
		if random is None: np.random.seed(42)
		else: np.random.seed(random)
		#Ensure Input Data Types are Correct
		assert(type(self.Ng) is int)
		assert(type(self.sigma0) in [float,int] and self.sigma0>0)
		assert(type(self.sigmaRel) in [float,int] and self.sigmaRel>=0 and self.sigmaRel<=self.sigma0)
		assert(type(self.zcmb_bounds) in [list,np.ndarray] and len(self.zcmb_bounds)==2)

		#If Sg or sigma_fit_s are individual numbers, expand out to create list for all galaxies
		for x in ['Sg','sigma_fit_s']:
			y = self.__dict__[x]
			if type(y) in [np.ndarray,list]:#If lists, ensure types in list are correct
				if x=='Sg':
					for sg in y:
						assert(type(sg) is int)
						assert(sg>=2)
				elif x=='sigma_fit_s':
					for sig in y:
						assert(sig>0)

			elif type(y) in [int,float]:#If int/floats, create list
				if x=='Sg':
					assert(type(y) is int)
					self.__dict__[x] = [y for _ in range(self.Ng)]
				if x=='sigma_fit_s':
					assert(y>0)
					self.__dict__[x] = [y for _ in range(self.Ng) for __ in range(self.Sg[_])]

		#Empty pandas df
		dfcolumns = ['Galaxy','SN','mus','mu_errs']
		if external_distances:
			dfcolumns += ['muext_zcmb_hats','zcmb_hats','zcmb_errs']
		dfmus = pd.DataFrame(columns=dfcolumns)

		#For each galaxy, simulate individual siblings distances, collate results into pandas df
		counter = -1
		for g in range(self.Ng):
			dM_Common   = np.random.normal(0,self.sigmaCommon)
			dM_Rels     = np.random.normal(0,self.sigmaRel,self.Sg[g])
			#zcmb        = np.random.uniform(self.zcmb_bounds[0],self.zcmb_bounds[1])
			#mu          = self.cosmo.distmod(zcmb).value
			zHD    = np.random.uniform(self.zcmb_bounds[0],self.zcmb_bounds[1])
			zcmb   = zHD  + np.random.normal(0,sigmapec/c_light)
			zhelio = zcmb + 0#zpo=0
			mu   = self.cosmo.distmod(zHD).value + 5*np.log10((1+zhelio)/(1+zHD))
			if external_distances:
				zcmb_hat = np.random.normal(zcmb, zcmberr)#np.sqrt(np.square(zcmberr) + np.square(sigmapec/c_light)))
				mu_ext   = self.cosmo.distmod(zcmb_hat).value#Simple estimate where we assume zcmb=zhelio

			for s in range(self.Sg[g]):
				counter +=1
				error_term = np.random.normal(0,self.sigma_fit_s[counter])
				mu_hat     = mu + dM_Common + dM_Rels[s] + error_term
				row        = [f'G{g+1}',f'S{s+1}_G{g+1}',mu_hat, self.sigma_fit_s[counter]] + external_distances*[mu_ext, zcmb_hat, zcmberr]
				dfmus      = dfmus._append(dict(zip(list(dfmus.columns),row)), ignore_index=True)

		#End product
		self.dfmus = dfmus
