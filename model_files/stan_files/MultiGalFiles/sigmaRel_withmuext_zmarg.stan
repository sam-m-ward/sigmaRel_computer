data {
    int<lower=0> Ng;
    array[Ng] int<lower=2> S_g;
    int <lower=Ng*2> Nsibs;

    vector[Nsibs] mu_sib_phots;
    vector[Nsibs] mu_sib_phot_errs;

    vector[Ng] zg_data;
    vector[Ng] zgerrs_data;
    real q0;
    real j0;
    real c_H0;

    real<lower=0,upper=1> sigmaRel_input;      //Option to fix sigmaRel at a fraction of sigma0, if 0, don't fix, if 1, fix
    real<lower=0,upper=1> eta_sigmaRel_input;  //The fraction of sigma0, e.g. eta_sigmaRel_input=0.5 is sigmaRel = sigma0/2

    //real<lower=0,upper=1> sigma0;    //Data
    //real<lower=0,upper=1> pec_unity; //Data
}

parameters {
    vector<lower=0,upper=1>[Ng] nuhelio;
    vector<lower=0>[Ng] zcmb;

    vector[Ng] eta_dM_common;
    vector[Nsibs] eta_dM_rel;

    real<lower=0,upper=1> sigma0;    //Model
    real<lower=0,upper=1> pec_unity; //Model
    real<lower=0,upper=1> eta_sigmaRel_param;
}

transformed parameters {
    vector[Ng] muLCDM;
    vector[Ng] dM_common;
    vector[Nsibs] dM_rel;
    vector[Nsibs] dM_sibs;
    real sigmaRel;
    vector[Ng] zhelio;

    if (sigmaRel_input!=0) {
      sigmaRel = eta_sigmaRel_input*sigma0;
    }
    else {
      sigmaRel = eta_sigmaRel_param*sigma0;
    }

    dM_rel    = eta_dM_rel*sigmaRel;
    dM_common = eta_dM_common*sqrt(square(sigma0)-square(sigmaRel));
    for (g in 1:Ng) {
      zhelio[g] = zcmb[g] - pec_unity * ( inv_Phi ( nuhelio[g] * Phi (zcmb[g]/pec_unity) ) );
      muLCDM[g] = 5*log10(c_H0*zcmb[g]*( 1+(1-q0)*zcmb[g]/2 - (1-q0-3*square(q0)+j0)*square(zcmb[g])/6 ) ) + 25 + 5*log10((1+zhelio[g])/(1+zcmb[g]));
      dM_sibs[sum(S_g[:g-1])+1:sum(S_g[:g])] = dM_rel[sum(S_g[:g-1])+1:sum(S_g[:g])] + dM_common[g];
    }
}

model {
    eta_dM_common ~ std_normal();
    eta_dM_rel    ~ std_normal();
    nuhelio       ~ uniform(0,1);

    eta_sigmaRel_param ~ uniform(0,1);
    sigma0    ~ uniform(0,1);
    pec_unity ~ uniform(0,1);

    for (g in 1:Ng){
      mu_sib_phots[sum(S_g[:g-1])+1:sum(S_g[:g])] ~ normal(muLCDM[g]+dM_sibs[sum(S_g[:g-1])+1:sum(S_g[:g])],mu_sib_phot_errs[sum(S_g[:g-1])+1:sum(S_g[:g])]);
    }
    zg_data ~ normal(zhelio, zgerrs_data);

}

generated quantities {
  real<lower=0> sigmapec;
  real<lower=0> sigmaCommon;
  real<lower=0,upper=1> rho;
  rho         = 1-square(sigmaRel)/square(sigma0);
  sigmaCommon = sqrt(square(sigma0)-square(sigmaRel));
  sigmapec    = pec_unity*299792.458;
}
