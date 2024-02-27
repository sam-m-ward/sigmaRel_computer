import sys

rootpath = './'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *
from simulate_distances import *

#simulator = SiblingsDistanceSimulator(Ng=100,Sg=3,external_distances=True)
#dfmus     = simulator.dfmus
#print (dfmus)
#multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True)
##multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[0.1,0.15,1.0])
#multigal.sigmaRel_sampler(sigma0='free',sigmapec=250,use_external_distances=True)
#multigal.plot_posterior_samples()
#'''
data = {    'Galaxy':['1316','1316','1316','3190','3190','5018','5018','1404','1404','MCG22427','MCG22427','1575','1575'],
            'SN':['1980N','1981D','2006dd','2002bo','2002cv','2002dj','2021fxy','2007on','2011iv','2011at','2020jgl','2020sjo','2020zhh'],
            #'mus':[31.753,31.759,31.466,32.408,33.155,33.187,32.978,32.043,31.343,32.813,32.669,35.780,35.596],
            #'mu_errs':[0.027,0.028,0.022,0.047,0.024,0.019,0.026,0.010,0.017,0.024,0.030,0.022,0.056]
            'mus':[31.387,31.389,31.201,32.186,31.902,33.148,32.977,31.502,31.207,32.724,32.676,35.830,35.552],
            'mu_errs':[0.080,0.073,0.044,0.124,0.022,0.044,0.031,0.032,0.053,0.0062,0.032,0.038,0.094]
}
dfmus = pd.DataFrame(data=data)
multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True)
multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0])
multigal.plot_all_distances()
#'''
