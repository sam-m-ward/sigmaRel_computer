import sys

rootpath = './'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *
#from simulate_distances import *
#simulator = SiblingsDistanceSimulator(Ng=100,Sg=3,external_distances=True)
#dfmus     = simulator.dfmus


#LOAD IN CHAINS
from glob import glob
chains_path = '../../bayesn_for_ztf/bayesn-pre-release-dev/ZTFtest3/ZTFtest3_Samples/'
#chains_path = '../../bayesn_for_ztf/bayesn-pre-release-dev/ZTFtest3_freeRV/ZTFtest3_freeRV_Samples/'
files = glob(chains_path+'*.npy')
snes  = {ff.split(chains_path)[1].split('_ZTF_DR2.snana_chains.npy')[0]:ff for ff in files}
dfmus = {'sn':[]} ; PARS = ['mu','AV','theta']
for sn,ff in snes.items():
    dfmus['sn'].append(sn)
    for PAR in PARS:
        if PAR not in dfmus.keys():
            dfmus[PAR], dfmus[PAR+'_errs'], dfmus[PAR+'_samps'] = [],[],[]
        x = np.load(ff,allow_pickle=True).item()
        if PAR=='mu':   dfmus[f'{PAR}_samps'].append(x[PAR]+x['delM'])
        else:           dfmus[f'{PAR}_samps'].append(x[PAR])
        dfmus[PAR].append(np.median(dfmus[f'{PAR}_samps'][-1]))
        dfmus[f'{PAR}_errs'].append(np.std(dfmus[f'{PAR}_samps'][-1]))
dfmus = pd.DataFrame(dfmus)
dfmus.set_index('sn',inplace=True)

#LOAD IN SUMMARIES
files = glob(chains_path+'*.pkl')
snes  = {ff.split(chains_path)[1].split('_ZTF_DR2.snana_summary.pkl')[0]:ff for ff in files}
Rhats = {}
Rhat_cols = [f'{PAR}_Rhats' for PAR in PARS]
for sn,ff in snes.items():
    with open(ff,'rb') as f:
        summary = pickle.load(f)
    dfmus.loc[sn,Rhat_cols] = summary.loc[PARS]['R_hat'].values

#LOAD IN SNE TO MAP TO GALAXY
sng = '../../bayesn_for_ztf/bayesn-pre-release-dev/bayesn-data/lcs/meta/ztf_dr2_siblings_redshifts.txt'
sng = pd.read_csv(sng,names=['sn','z'],sep='\s')
sn_to_g = {sn:1+isn//2 for isn,sn in enumerate(sng.sn)}


#COMBINE CHAINS AND SIBS GALAXIES
dfmus = dfmus.loc[sn_to_g.keys()]
dfmus['Galaxy'] = list(sn_to_g.values())
data = {
    'Galaxy':list(sn_to_g.values()),
    'SN':list(sn_to_g.keys()),
}
for PAR in ['mu','AV','theta']:
    data[f'{PAR}s']      = [dfmus[PAR].loc[sn] for sn in sn_to_g]
    data[f'{PAR}_errs']  = [dfmus[f'{PAR}_errs'].loc[sn] for sn in sn_to_g]
    data[f'{PAR}_samps'] = [dfmus[f'{PAR}_samps'].loc[sn] for sn in sn_to_g]
    data[f'{PAR}_Rhats'] = [dfmus[f'{PAR}_Rhats'].loc[sn] for sn in sn_to_g]

dfmus = pd.DataFrame(data=data)
print (dfmus.columns)

#APPLY FILTERING
#dfmus = dfmus[~dfmus['Galaxy'].isin([10,11])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([3,5,10,11])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([5,10,11])]
#dfmus = dfmus[~dfmus['Galaxy'].isin([2,5,10,11])]

#ANALYSE
multigal = multi_galaxy_siblings(dfmus,sigma0='free',sigmapec=250,use_external_distances=True)
#multigal.print_table()
multigal.compute_analytic_multi_gal_sigmaRel_posterior(prior_upper_bounds=[1.0],blind=True)
multigal.plot_parameters(['mu','AV','theta'])
#'''
