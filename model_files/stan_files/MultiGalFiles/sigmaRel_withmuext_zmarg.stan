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

    real<lower=0,upper=1> sigmaRel_input;      //Option to fix sigmaRel at a fraction of sigma0, if 0, don't fix, if 1, fix
    real<lower=0,upper=1> eta_sigmaRel_input;  //The fraction of sigma0, e.g. eta_sigmaRel_input=0.5 is sigmaRel = sigma0/2

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

    real<lower=0,upper=1> sigma0;    //Model
    real<lower=0,upper=1> pec_unity; //Model
    real<lower=0,upper=1> eta_sigmaRel_param;
}

transformed parameters {
    vector[Ng] muLCDM;
    vector[Ng] zhelios;
    vector[Ng] dM_common;
    vector[Nsibs] dM_rel;
    vector[Nsibs] dM_sibs;
    real sigmaRel;
    vector[Nsibs] dmuLCDM;

    if (sigmaRel_input!=0) {
      sigmaRel = eta_sigmaRel_input*sigma0;
    }
    else {
      sigmaRel = eta_sigmaRel_param*sigma0;
    }

    zhelios   = zcosmos + pec_unity*eta_pecs + zpo_hats;
    dM_rel    = eta_dM_rel*sigmaRel;
    dM_common = eta_dM_common*sqrt(square(sigma0)-square(sigmaRel));
    dmuLCDM   = Q_sib_zhel*alpha_zhel;
    for (g in 1:Ng) {
      muLCDM[g] = 5*log10(c_H0*zcosmos[g]*( 1+(1-q0)*zcosmos[g]/2 - (1-q0-3*square(q0)+j0)*square(zcosmos[g])/6 ) ) + 25 + 5*log10((1+zhelios[g])/(1+zcosmos[g]));
      dM_sibs[sum(S_g[:g-1])+1:sum(S_g[:g])] = dM_rel[sum(S_g[:g-1])+1:sum(S_g[:g])] + dM_common[g];
      dmuLCDM[sum(S_g[:g-1])+1:sum(S_g[:g])] *= (zhelios[g] - zhelio_hats[g]); //Additional term for marginalising over photometric distance dependence on heliocentric redshift
    }
}

model {
    eta_dM_common ~ std_normal();
    eta_dM_rel    ~ std_normal();
    eta_pecs      ~ std_normal();
    //zhelios ~ normal(zcosmos+zpo_hats, pec_unity); //Alternative form less stable

    eta_sigmaRel_param ~ uniform(0,1);
    sigma0    ~ uniform(0,1);
    pec_unity ~ uniform(0,1);

    for (g in 1:Ng){
      mu_sib_phots[sum(S_g[:g-1])+1:sum(S_g[:g])] ~ normal(muLCDM[g]-dmuLCDM[sum(S_g[:g-1])+1:sum(S_g[:g])]+dM_sibs[sum(S_g[:g-1])+1:sum(S_g[:g])],mu_sib_phot_errs[sum(S_g[:g-1])+1:sum(S_g[:g])]);
    }
    zhelio_hats ~ normal(zhelios, zhelio_errs);

}

generated quantities {
  real<lower=0> sigmapec;
  real<lower=0> sigmaCommon;
  real<lower=0,upper=1> rho;
  rho         = 1-square(sigmaRel)/square(sigma0);
  sigmaCommon = sqrt(square(sigma0)-square(sigmaRel));
  sigmapec    = pec_unity*299792.458;
}
