import sys
rootpath = './'
sys.path.append(rootpath+'model_files/')

from load_data import *
from ResModel import *


dfmus = load_dfmus('ZTFtest5')
#dfmus = dfmus.iloc[::2].copy()

#dfmus = dfmus[~dfmus['Galaxy'].isin([3,4,11,12])] #Limit to cosmology sample
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,4,6,11,12])] #Limit to cosmology sample


rml = ResModelLoader(dfmus)
rml.plot_res()
rml.sample_posterior(pxs=['theta'],beta=[0.14])
rml.plot_posterior_samples()
