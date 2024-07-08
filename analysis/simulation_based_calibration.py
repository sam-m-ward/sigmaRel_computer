import sys

rootpath = '../'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *

arcsine_prior_lines = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim \textrm{Arcsine}(0,1)$"]
priorC_lines        = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim U(0,1)$"]

from simulate_distances import *

Ng = 100 ; Sg = 2 ; RS=10
tick_cross = dict(zip([True,False],['\\xmark','\\cmark']))
Summary_Strs = []
for rs in np.arange(RS):
#for rs in [9]:
    for sigR,RHO in zip([0,0.1/((2)**0.5),0.1],[1,0.5,0]):
        print ((('#~#~#'*30)+'\n')*3)
        print (f'Performing: random_state={rs}/{RS}; sigR={sigR} equiv rho={RHO}')
        print ((('#-#-#'*30)+'\n')*3)
        simulator = SiblingsDistanceSimulator(Ng=Ng,Sg=Sg,external_distances=True,sigmaRel=sigR,zcmberr=1e-5,random=42+rs)
        dfmus     = simulator.dfmus
        dfmus['zhelio_hats'] = dfmus['zcmb_hats']
        dfmus['zhelio_errs'] = dfmus['zcmb_errs']

        samplename = f'Ng{Ng}_Sg{Sg}_Rs{rs}_Truesigma0{simulator.sigma0}_TruesigmaRel{round(simulator.sigmaRel,3)}'
        multigal = multi_galaxy_siblings(dfmus,samplename=samplename,sigma0=1.0,eta_sigmaRel_input=None,sigmapec=250,use_external_distances=False,rootpath=rootpath)
        multigal.n_warmup = 250 ; multigal.n_sampling = 250

        #Multi-model overlay
        zcosmo = 'zcmb'
        sigma0 = 'free'
        Rhat_threshold = 1.02 ; Ntrials=10
        overwrite=True
        overwrite=False
        savekey   = f'multigalsims_{samplename}_Modelsigma0{sigma0}'

        multigal.sigmaRel_sampler(sigma0=sigma0,sigmapec=250,use_external_distances=True,zcosmo=zcosmo,overwrite=overwrite,Rhat_threshold=Rhat_threshold, Ntrials=Ntrials)
