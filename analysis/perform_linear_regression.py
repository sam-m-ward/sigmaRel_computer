import sys
rootpath = '../'
sys.path.append(rootpath+'model_files/')

from load_data import *
from ResModel import *
from sigmaRel import *


dfmus    = load_dfmus('ZTFtest5')
multigal = multi_galaxy_siblings(dfmus,rootpath=rootpath)
#multigal.trim_sample()


rml = ResModelLoader(multigal.dfmus,rootpath=rootpath)
rml.plot_res(p1='theta')
rml.sample_posterior(pxs=['theta'])#,beta=[0.102])#.14])
rml.plot_posterior_samples()
