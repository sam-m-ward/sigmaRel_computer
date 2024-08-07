import sys
rootpath = '../../'
sys.path.append(rootpath+'model_files/')

from load_data import *
from ResModel import *
from sigmaRel import *


dfmus    = load_dfmus('ZTFtest5',rootpath=rootpath)
multigal = multi_galaxy_siblings(dfmus,rootpath=rootpath)
#multigal.trim_sample()
multigal.plot_parameters(PAR=['mu','AV','theta'])#,subtract_g_mean=True)


rml = ResModelLoader(multigal.dfmus,rootpath=rootpath)

'''#Plots
rml.plot_res_pairs(px='theta')
rml.plot_res_pairs(px='AV')
rml.plot_res_pairs(px='theta',py='AV')
rml.plot_res(px='theta')
rml.plot_res(px='AV')
rml.plot_res(px='etaAV')
rml.plot_res(px='theta',py='AV')
rml.plot_res(px='theta',py='etaAV')
rml.plot_res(px='theta',py='AV',zerox=False,zeroy=False)
rml.plot_res(px='theta',py='AV',zerox=False)
rml.plot_res(px='theta',py='AV',zeroy=False)
rml.plot_res(px='theta',zerox=False)
rml.plot_res(px='AV',zerox=False)
#'''
err=1/0

rml.sample_posterior(pxs=['theta'])#,beta=[0.102])#.14])
rml.plot_posterior_samples()
