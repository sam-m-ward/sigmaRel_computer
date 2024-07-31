from glob import glob
import numpy as np
import pandas as pd
from scipy.special import erfinv
import pickle
from contextlib import suppress
from astropy.cosmology import FlatLambdaCDM
from sklearn.linear_model import LinearRegression
import json

from numpy import pi, sin, cos, arctan2, hypot
def heliocorr(zhel, RA, Dec):
    """
    A function to transform redshifts from the heliocentric-frame to the
    CMB-frame using Planck (2018) CMB Dipole measurements.

    Inputs:
        zhel: float or numpy array, input heliocentric redshift(s)
        RA: float or numpy array, object equatorial right ascension(s)
        Dec: float or numpy array, object equatorial declination(s)

    Outputs:
        zcmb: numpy array, redshift(s) corrected to the CMB frame
        alpha: float or numpy array, angular separation from CMB dipole (rad)
    """

    v_Sun_Planck = 369.82  # +/- 0.11 km/s
    l_dipole_Planck = 264.021  # +/- 0.011 deg
    b_dipole_Planck = 48.253  # +/- 0.005 deg
    c = 299792.458  # km/s

    # Co-ords of North Galactic Pole (ICRS): RA = 192.729 ± 0.035 deg, Dec = 27.084 ± 0.023 deg
    # (https://doi.org/10.1093/mnras/stw2772)
    # Co-ords of Galactic Centre (ICRS): RA = 17h45m40.0409s, Dec = −29d00m28.118s (see above reference)
    #                                    RA = 266.41683708 deg, Dec = -29.00781056 deg
    # Ascending node of the galactic plane = arccos(sin(Dec_GC)*cos(Dec_NGP)-cos(Dec_GC)*sin(Dec_NGP)*cos(RA_NGP-RA_GC))
    #                                      = 122.92828126730255 = l_0
    # Transform CMB dipole from (l,b) to (RA,Dec):
    #     Dec = arcsin(sin(Dec_NGP)*sin(b)+cos(Dec_NGP)*cos(b)*cos(l_0-l))
    #         = -6.9895105228347 deg
    #     RA = RA_NGP + arctan((cos(b)*sin(l_0-l)) / (cos(Dec_NGP)*sin(b)-sin(Dec_NGP)*cos(b)*cos(l_0-l)))
    #        = 167.81671014708002 deg

    # Astropy co-ordinates are old and low precision:
    # RA_NGP_J2000 = 192.8594812065348, Dec_NGP_J2000 = 27.12825118085622, which are converted from B1950
    # RA_NGP_B1950 = 192.25, Dec_NGP_B1950 = 27.4
    # l_0_B1950 = 123
    # l_0_J2000 = 122.9319185680026
    # Introduces around 1e-6 error in redshift

    RA_Sun_Planck = 167.816710  # deg
    Dec_Sun_Planck = -6.989510  # deg

    rad = pi / 180.0
    # using Vincenty formula because it is more accurate
    alpha = arctan2(
        hypot(
            cos(Dec_Sun_Planck * rad) * sin(np.fabs(RA - RA_Sun_Planck) * rad),
            cos(Dec * rad) * sin(Dec_Sun_Planck * rad)
            - sin(Dec * rad)
            * cos(Dec_Sun_Planck * rad)
            * cos(np.fabs(RA - RA_Sun_Planck) * rad),
        ),
        sin(Dec * rad) * sin(Dec_Sun_Planck * rad)
        + cos(Dec * rad)
        * cos(Dec_Sun_Planck * rad)
        * cos(np.fabs(RA - RA_Sun_Planck) * rad),
    )

    v_Sun_proj = v_Sun_Planck * np.cos(alpha)

    z_Sun = np.sqrt((1.0 + (-v_Sun_proj) / c) / (1.0 - (-v_Sun_proj) / c)) - 1.0
    # Full special rel. correction since it is a peculiar vel

    min_z = 0.0

    zcmb = np.where(zhel > min_z, (1 + zhel) / (1 + z_Sun) - 1, zhel)

    return zcmb#, alpha

from astropy import units as u
from astropy.coordinates import SkyCoord
#https://cosmicflows.iap.fr/table_query/
def get_l_b(RA, Dec):#For cosmic flows query
    coord = SkyCoord(ra=RA*u.degree, dec=Dec*u.degree, frame='icrs')
    l = coord.galactic.l.value
    b = coord.galactic.b.value
    return l,b

def load_dfmus(chains_file='ZTFtest5',rootpath='../', tau=0.252):
    #LOAD IN POSTERIOR CHAINS AND CREATE DFMUS
    chains_path = f'{rootpath}../../bayesn_for_ztf/bayesn-pre-release-dev/{chains_file}/{chains_file}_Samples/'
    files = glob(chains_path+'*.npy')
    snes  = {ff.split(chains_path)[1].split('_ZTF_DR2.snana_chains.npy')[0]:ff for ff in files}
    dfmus = {'SN':[]} ; PARS = ['mu','AV','theta']
    for sn,ff in snes.items():
        dfmus['SN'].append(sn)
        for PAR in PARS:
            if f'{PAR}s' not in dfmus.keys():
                dfmus[f'{PAR}s'], dfmus[f'{PAR}_errs'], dfmus[f'{PAR}_samps'] = [],[],[]
            x = np.load(ff,allow_pickle=True).item()
            if PAR=='mu':   dfmus[f'{PAR}_samps'].append(x[PAR]+x['delM'])
            else:           dfmus[f'{PAR}_samps'].append(x[PAR])
            dfmus[f'{PAR}s'].append(np.median(dfmus[f'{PAR}_samps'][-1]))
            dfmus[f'{PAR}_errs'].append(np.std(dfmus[f'{PAR}_samps'][-1]))
    dfmus = pd.DataFrame(dfmus)
    dfmus.set_index('SN',inplace=True)

    dfmus['etaAV_samps'] = dfmus['AV_samps'].apply(lambda x : np.sqrt(2)*erfinv(1-2*np.exp(-x/tau)))
    dfmus['etaAVs']      = dfmus['etaAV_samps'].apply(np.median)
    dfmus['etaAV_errs']  = dfmus['etaAV_samps'].apply(np.std)

    #LOAD IN POSTERIOR RHATS AND ADD TO DFMUS
    files = glob(chains_path+'*.pkl')
    snes  = {ff.split(chains_path)[1].split('_ZTF_DR2.snana_summary.pkl')[0]:ff for ff in files}
    Rhats = {}
    Rhat_cols = [f'{PAR}_Rhats' for PAR in PARS]
    for sn,ff in snes.items():
        with open(ff,'rb') as f:
            summary = pickle.load(f)
        dfmus.loc[sn,Rhat_cols] = summary.loc[PARS]['R_hat'].values

    #LOAD IN SNE TO MAP SIBLINGS TO GALAXY ID
    sng = f'{rootpath}../../bayesn_for_ztf/bayesn-pre-release-dev/bayesn-data/lcs/meta/ztf_dr2_siblings_redshifts.txt'
    sng = pd.read_csv(sng,names=['SN','zhelio_hats'],sep='\s')
    sn_to_g = {sn:1+isn//2 for isn,sn in enumerate(sng.SN)}

    #ADD GALAXY IDS TO DFMUS
    dfmus = dfmus.loc[sn_to_g.keys()]
    dfmus['Galaxy'] = list(sn_to_g.values())

    #GET ZHELIOS, CONVERT TO ZCMBS, AND ADD TO DFMUS
    #Get zhelio errors using last decimal place
    def dec_place(x):
        for _ in range(10):
            if round((x*(10**(_)))%1,10)==0:
                break
        return 10**(-_)
    sng.set_index('SN',inplace=True)
    cosmo  = FlatLambdaCDM(H0=73.24,Om0=0.28)
    sng['zhelio_errs'] = sng['zhelio_hats'].apply(dec_place)

    #Get ra, dec
    df_radeczhel = pd.read_csv(f'{rootpath}../../bayesn_for_ztf/bayesn-pre-release-dev/bayesn-data/lcs/meta/ztf_dr2_RADec_zhelio_roughestimates.txt',names=['SN','ra','dec','z_helio'],sep='\s')
    df_radeczhel.set_index('SN',inplace=True)
    df_radeczhel.loc['ZTF20abmarcv_1'] = df_radeczhel.loc['ZTF20abmarcv'].iloc[0]#Two siblings on same pixel, introduce as new SNe
    df_radeczhel.loc['ZTF20abmarcv_2'] = df_radeczhel.loc['ZTF20abmarcv'].iloc[0]
    #print (df_radeczhel)
    sng = sng.merge(df_radeczhel[['ra','dec']],left_index=True,right_index=True,how='left').loc[sng.index]
    #print (sng)
    #Add to dfmus, then compute zcmbs
    dfmus = dfmus.merge(sng[['zhelio_hats','zhelio_errs','ra','dec']],left_index=True,right_index=True,how='left').loc[dfmus.index]
    #print (dfmus)
    dfmus['zcmb_hats']  = dfmus[['zhelio_hats','ra','dec']].apply(lambda x: heliocorr(x[0],x[1],x[2]),axis=1)#zcmb by correcting zhelio for cmb dipole

    #Get text file for use in cosmic flows
    try:
        #df_flowcorr = pd.read_csv('products/table_for_cosmicflows_out.txt',skiprows=17,header=None,names=['long','lat','lg','bg','vcmb','x','DL','DA','vHD','vpec'],sep='\s+')
        df_flowcorr = pd.read_csv(f'{rootpath}products/table_for_cosmicflows_v_out.txt',skiprows=17,header=None,names=['long','lat','lg','bg','vcmb','x','DL','DA','vHD','vpec'],sep='\s+')
        assert(df_flowcorr.shape[0]==dfmus.shape[0])
        dfmus.loc[:,'zHD_hats'] = df_flowcorr['vHD'].values/299792.458
        dfmus.loc[:,'mu_cosmicflows'] = 5*np.log10(df_flowcorr['DL'].values)+25
    except:
        #dfmus[['l','b']] = dfmus[['ra','dec','zcmb_hats']].apply(lambda x: get_l_b(x[0],x[1]),axis=1,result_type='expand')
        #dfmus[['l','b','zcmb_hats']].to_csv('products/table_for_cosmicflows.txt',index=False,header=None,sep=' ')
        #dfmus[['ra','dec','zcmb_hats']].to_csv('products/table_for_cosmicflows.txt',index=False,header=None,sep=' ')
        dfmus['vcmb_hats'] = dfmus['zcmb_hats']*299792.458
        dfmus[['ra','dec','vcmb_hats']].to_csv(f'{rootpath}products/table_for_cosmicflows_v.txt',index=False,header=None,sep=' ')
        raise Exception('Please take products/table_for_cosmicflows.txt and upload to https://cosmicflows.iap.fr/table_query/ using J2000 galactic coords and velocity in c \n save as products/table_for_cosmicflows_out.txt')

    #For simplicity, take mean of zcmb and zHD (and zhelio is already same for both sibs)
    #print ('Taking mean of zHD, differences between original and new should be negligible:')
    #print (dfmus['zHD_hats'] - dfmus.groupby('Galaxy')['zHD_hats'].transform('mean'))
    dfmus['zcmb_hats']  = dfmus.groupby('Galaxy')['zcmb_hats'].transform('mean')#Simply take the mean in each Galaxy
    dfmus['zHD_hats']   = dfmus.groupby('Galaxy')['zHD_hats'].transform('mean')#Simply take the mean in each Galaxy
    #dfmus['muext_hats'] = dfmus[['zhelio_hats','zHD_hats']].apply(lambda z: cosmo.distmod(z[1]).value + 5*np.log10((1+z[0])/(1+z[1])),axis=1)
    #########################
    '''
    #Examine correlation of zhelio and zcmbs, simulate gaussian zhels, compute zcmbs, then compute their correlation
    Nsamps=10000
    zhel_samps = dfmus.groupby('Galaxy')[['zhelio_hats','zhelio_errs']].apply(lambda x: list(np.random.normal(x.iloc[0]['zhelio_hats'],x.iloc[0]['zhelio_errs'],Nsamps))).to_frame('zhel_samps')
    dfmus = dfmus.merge(zhel_samps,on='Galaxy',how='left').set_index(dfmus.index)
    dfmus['zcmb_samps'] = dfmus[['zhel_samps','ra','dec']].apply(lambda x: heliocorr(np.asarray(x[0]),x[1],x[2]),axis=1)
    dfmus['zcmb_errs']  = dfmus['zcmb_samps'].apply(np.std)
    def get_cov(x):
        x = np.array([np.asarray(xi) for xi in x])
        return np.corrcoef(x)[1,0]
    dfmus['z_hel_cmb_cov'] = dfmus[['zhel_samps','zcmb_samps']].apply(get_cov,axis=1)
    #'''
    #########################
    #Get mu vs. zhelio alpha-slope estimates from zhelio grid; linear OLS fit
    dfmus.loc[:,'alpha_mu_z'] = 0
    try:
        dzhel_to_load = [0.01]
        for dzh in dzhel_to_load:
            df_zgrid = pd.read_csv(f'{rootpath}products/sens_to_zhel/mu_dz{dzh}.csv').set_index('sn',drop=True)
            dzhels = list(df_zgrid.columns)
            for sn in df_zgrid.index:
                ys = []
                for iz,dzhel in enumerate(dzhels):
                    mu  = float(json.loads(df_zgrid.loc[sn][dzhel])[0])
                    mu0 = float(json.loads(df_zgrid.loc[sn]['0.0'])[0])
                    ys.append(mu-mu0)

                mo = LinearRegression(fit_intercept=False)
                mo.fit(np.asarray(dzhels).astype(float).reshape(-1,1),ys)
                mhat = mo.coef_[0]
                dfmus.loc[sn,'alpha_mu_z']  = round(mhat,5)
    except Exception as ee:
        print (f'No photometric distances grid in zhelio and/or exception: {ee}')
    #########################

    #FINAL TOUCHES
    print ('Setting measurement errors on zcmb/zHD to be same as on zhelio, i.e. the sigmaz term used in mu-pipeline cosmo-distance errors; thus assuming errors are perfectly correlated and no additional errors in CMB or flow corrections')
    print ('This is fine for zcmb, but unclear how flow correction would change with strong changes in zhelio(or equivalently zcmb)')
    dfmus['zcmb_errs'] = dfmus['zhelio_errs'].copy()#Keep things simple seeing as these columns are essentially identical
    dfmus['zHD_errs']  = dfmus['zhelio_errs'].copy()#Keep things simple seeing as these columns are essentially identical
    dfmus.reset_index(inplace=True)#Move SN to columns
    return dfmus
    '''
    #dfmus = pd.DataFrame(data=data) #dfmus['mudiff'] = dfmus['mus']-dfmus['muext_hats']#print (dfmus[['mus','muext_hats','zcmb_hats','mudiff']])
    def mean_mapper(x):
        mean_samps = x.mean(axis=0)
        x = x.apply(lambda x: x-mean_samps)
        return x
    for PAR in ['mu','AV','theta','etaAV']:
        dfmus[f'{PAR}_res']      = dfmus[f'{PAR}s'] - dfmus.groupby('Galaxy')[f'{PAR}s'].transform('mean')
        dfmus[f'{PAR}res_samps'] = dfmus.groupby('Galaxy')[f'{PAR}_samps'].transform(lambda x: mean_mapper(x))
    '''
