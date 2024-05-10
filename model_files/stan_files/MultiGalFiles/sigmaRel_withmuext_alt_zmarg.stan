data {
    int<lower=0> Ng;
    array[Ng] int<lower=2> S_g;
    int <lower=Ng*2> Nsibs;

    vector[Nsibs] mu_sib_phots;
    vector[Nsibs] mu_sib_phot_errs;

    vector[Ng] zhelio_hats;//zhelio point estimates and errors
    vector[Ng] zhelio_errs;
    vector[Ng] zpo_hats;   //observer peculiar velocity offsets: zh = zcosmo + zpo

    real q0;
    real j0;
    real c_H0;

    int<lower=0> Nzhel;                //No. of galaxies with large heliocentric redshift errors (e.g. sigma_zhelio = 0.01)
    vector[Nzhel] alpha_zhel;          //Slopes of dmu_phot = alpha_zhel*dzhelio
    matrix[Nsibs,Nzhel] Q_sib_zhel;    //Transformation of Nzhel siblings to full vector of siblings

    //real<lower=0,upper=1> sigma0;    //Data
    //real<lower=0,upper=1> pec_unity; //Data
}

parameters {
    vector<lower=0>[Ng] zcosmos;
    vector[Ng] eta_pecs;

    vector[Ng] eta_dM_common;
    vector[Nsibs] eta_dM_rel;

    real<lower=0,upper=1> sigmaRel;   //Model
    real<lower=0,upper=1> sigmaCommon;//Model
    real<lower=0,upper=1> pec_unity; //Model
}

transformed parameters {
    vector[Ng] muLCDM;
    vector[Ng] zhelios;
    vector[Ng] dM_common;
    vector[Nsibs] dM_rel;
    vector[Nsibs] dM_sibs;
    vector[Nsibs] dmuLCDM;

    zhelios   = zcosmos + pec_unity*eta_pecs + zpo_hats;
    dM_rel    = eta_dM_rel*sigmaRel;
    dM_common = eta_dM_common*sigmaCommon;
    dmuLCDM   = Q_sib_zhel*alpha_zhel;

    for (g in 1:Ng) {
      muLCDM[g] = 5*log10(c_H0*zcosmos[g]*( 1+(1-q0)*zcosmos[g]/2 - (1-q0-3*square(q0)+j0)*square(zcosmos[g])/6 ) ) + 25 + 5*log10((1+zhelios[g])/(1+zcosmos[g]));
      dM_sibs[sum(S_g[:g-1])+1:sum(S_g[:g])]  = dM_rel[sum(S_g[:g-1])+1:sum(S_g[:g])] + dM_common[g];
      dmuLCDM[sum(S_g[:g-1])+1:sum(S_g[:g])] *= (zhelios[g] - zhelio_hats[g]); //Additional term for marginalising over photometric distance dependence on heliocentric redshift
    }
}

model {
    eta_dM_common ~ std_normal();
    eta_dM_rel    ~ std_normal();
    eta_pecs      ~ std_normal();
    //zhelios ~ normal(zcosmos+zpo_hats, pec_unity); //Alternative form less stable

    sigmaRel    ~ uniform(0,1);
    sigmaCommon ~ uniform(0,1);
    pec_unity   ~ uniform(0,1);

    for (g in 1:Ng){
      mu_sib_phots[sum(S_g[:g-1])+1:sum(S_g[:g])] ~ normal(muLCDM[g]-dmuLCDM[sum(S_g[:g-1])+1:sum(S_g[:g])]+dM_sibs[sum(S_g[:g-1])+1:sum(S_g[:g])],mu_sib_phot_errs[sum(S_g[:g-1])+1:sum(S_g[:g])]);
    }
    zhelio_hats ~ normal(zhelios, zhelio_errs);

}

generated quantities {
  real<lower=0> sigmapec;
  real<lower=0,upper=1> rho;
  real<lower=0> sigma0;

  sigma0 = sqrt(square(sigmaRel)+square(sigmaCommon));
  rho    = square(sigmaCommon)/square(sigma0);
  sigmapec = pec_unity*299792.458;
}
