import sys

rootpath = './'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *


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
