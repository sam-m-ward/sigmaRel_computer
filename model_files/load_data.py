from glob import glob
import numpy as np
import pandas as pd
from scipy.special import erfinv
import pickle
from contextlib import suppress
from astropy.cosmology import FlatLambdaCDM

def load_dfmus(chains_file='ZTFtest5', tau=0.252):
    #LOAD IN CHAINS
    chains_path = f'../../bayesn_for_ztf/bayesn-pre-release-dev/{chains_file}/{chains_file}_Samples/'
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

    dfmus['etaAV_samps'] = dfmus['AV_samps'].apply(lambda x : np.sqrt(2)*erfinv(1-2*np.exp(-x/tau)))
    dfmus['etaAV']       = dfmus['etaAV_samps'].apply(np.median)
    dfmus['etaAV_errs']  = dfmus['etaAV_samps'].apply(np.std)

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
    for PAR in ['mu','AV','theta','etaAV']:
        data[f'{PAR}s']      = [dfmus[PAR].loc[sn] for sn in sn_to_g]
        data[f'{PAR}_errs']  = [dfmus[f'{PAR}_errs'].loc[sn] for sn in sn_to_g]
        data[f'{PAR}_samps'] = [dfmus[f'{PAR}_samps'].loc[sn] for sn in sn_to_g]
        with suppress(KeyError):
            data[f'{PAR}_Rhats'] = [dfmus[f'{PAR}_Rhats'].loc[sn] for sn in sn_to_g]

    #Get redshifts
    def dec_place(x):
        for _ in range(10):
            if round((x*(10**(_)))%1,10)==0:
                break
        return 10**(-_)
    sng.set_index('sn',inplace=True)
    cosmo  = FlatLambdaCDM(H0=73.24,Om0=0.28)
    data['zcmb_hats']  = [sng['z'].loc[sn]  for sn in sn_to_g]
    data['zcmb_errs']  = [dec_place(z)            for z in data['zcmb_hats']]
    data['muext_hats'] = [cosmo.distmod(z).value  for z in data['zcmb_hats']]

    dfmus = pd.DataFrame(data=data) #dfmus['mudiff'] = dfmus['mus']-dfmus['muext_hats']#print (dfmus[['mus','muext_hats','zcmb_hats','mudiff']])

    '''
    def mean_mapper(x):
        mean_samps = x.mean(axis=0)
        x = x.apply(lambda x: x-mean_samps)
        return x
    for PAR in ['mu','AV','theta','etaAV']:
        dfmus[f'{PAR}_res']      = dfmus[f'{PAR}s'] - dfmus.groupby('Galaxy')[f'{PAR}s'].transform('mean')
        dfmus[f'{PAR}res_samps'] = dfmus.groupby('Galaxy')[f'{PAR}_samps'].transform(lambda x: mean_mapper(x))
    '''

    return dfmus
