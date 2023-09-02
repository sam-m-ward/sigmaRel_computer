import sys

rootpath = './'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *
from simulate_distances import *


simulator = SiblingsDistanceSimulator(Ng=30,Sg=2)
dfmus     = simulator.dfmus

sibs = multi_galaxy_siblings(dfmus)
#sibs.loop_single_galaxy_analyses()
sibs.compute_analytic_multi_gal_sigmaRel_posterior()

'''
for gal in dfmus['Galaxy'].unique():
	dfgal  = dfmus[dfmus['Galaxy']==gal]
	sibgal = siblings_galaxy(dfgal['mus'].values,dfgal['mu_errs'].values,dfgal['SN'].values,gal,sigma0=0.1)
	sibgal.plot_sigmaRel_posteriors()
	sibgal.plot_individual_distances()
	sibgal.combine_individual_distances()
	sibgal.plot_common_distances()
'''












#NGC 3147
'''
DATA = {
	#'RVfixed':{ 'mus': np.array([33.142,33.135,33.224]),
	#            'errors':np.array([0.053,0.065,0.070])},
	'RVfree' :{ 'mus': np.array([33.131,33.127,33.256]),
				'errors':np.array([0.071,0.128,0.198]),
				'names':['2021hpr','1997bq','2008fv'],
                'galname':'NGC3147'}
				}
for MODE in DATA:
	sibgal = siblings_galaxy(DATA[MODE]['mus'],DATA[MODE]['errors'],DATA[MODE]['names'],DATA[MODE]['galname'])
	sibgal.plot_sigmaRel_posteriors()
	sibgal.plot_individual_distances()
	sibgal.combine_individual_distances()
	sibgal.plot_common_distances()
'''
