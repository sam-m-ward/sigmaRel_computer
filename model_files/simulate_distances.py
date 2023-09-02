"""
Simulate Distances

Module contains class for simulating individual siblings distance estimates

Contains:
--------------------
SiblingsDistanceSimulator class:
	inputs: Ngal=30, Sg = 2, sigma0=0.1, sigmaRel=0.05, mu_bounds = [25,35], sigma_fit_s=0.05
--------------------

Written by Sam M. Ward: smw92@cam.ac.uk
"""

import numpy as np
import pandas as pd
import copy

class SiblingsDistanceSimulator:

    def __init__(self, Ngal=30, Sg = 2, sigma0=0.1, sigmaRel=0.05, mu_bounds = [25,35], sigma_fit_s=0.05):
        """
        Initialisation

        Parameters
        ----------
        Ngal : int
            Number of siblings galaxies

        Sg : int or list/array of ints
            Number of siblings per galaxy

        sigma0 : float>0
            total intrinsic scatter

        sigmaRel : float>0 and <sigma0
            relative intrinsic scatter

        mu_bounds : list/array of x2 floats, each>0
            random distances drawn from [mu_low, mu_high]

        sigma_fit_s : list/array of floats, each>0, or float
            the fitting uncertainties, i.e. measurement errors, on each individual siblings distance estimate

        End Product(s)
        ----------
        self.dfmus : pandas df
            contains the galaxy ID's and siblings, each with their individual siblings distance estimates
        """
        self.__dict__.update(locals())
        self.sigmaCommon = np.sqrt(np.square(self.sigma0)-np.square(self.sigmaRel))

        #Ensure Input Data Types are Correct
        assert(type(self.Ngal) is int)
        assert(type(self.sigma0) in [float,int] and self.sigma0>0)
        assert(type(self.sigmaRel) in [float,int] and self.sigmaRel>=0 and self.sigmaRel<=self.sigma0)
        assert(type(self.mu_bounds) is list and len(self.mu_bounds)==2)

        #If Sg or sigma_fit_s are individual numbers, expand out to create list for all galaxies
        for x in ['Sg','sigma_fit_s']:
            y = self.__dict__[x]
            if type(y) in [np.ndarray,list]:#If lists, ensure types in list are correct
                if x=='Sg':
                    for sg in y:
                        assert(type(sg) is int)
                elif x=='sigma_fit_s':
                    for sig in y:
                        assert(sig>0)

            elif type(y) in [int,float]:#If int/floats, create list
                if x=='Sg':
                    assert(type(y) is int)
                    self.__dict__[x] = [y for _ in range(self.Ngal)]
                if x=='sigma_fit_s':
                    assert(y>0)
                    self.__dict__[x] = [y for _ in range(self.Ngal) for __ in range(self.Sg[_])]

        #Empty pandas df
        dfmus = pd.DataFrame(columns=['Galaxy','SN','mu','mu_err'])

        #For each galaxy, simulate individual siblings distances, collate results into pandas df
        counter = -1
        for g in range(self.Ngal):
            dM_Common   = np.random.normal(0,self.sigmaCommon)
            dM_Rels     = np.random.normal(0,self.sigmaRel,self.Sg[g])
            mu          = np.random.uniform(self.mu_bounds[0],self.mu_bounds[1])
            for s in range(self.Sg[g]):
                counter +=1
                error_term = np.random.normal(0,self.sigma_fit_s[counter])
                mu_hat     = mu + dM_Common + dM_Rels[s] + error_term
                row        = [f'G{g+1}',f'S{s+1}_G{g+1}',mu_hat, self.sigma_fit_s[counter]]
                dfmus      = dfmus._append(dict(zip(list(dfmus.columns),row)), ignore_index=True)

        #End product
        self.dfmus = dfmus
