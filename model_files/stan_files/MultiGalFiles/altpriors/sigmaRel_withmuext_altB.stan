data {
    int<lower=0> Ng;
    array[Ng] int<lower=2> S_g;
    int <lower=Ng*2> Nsibs;

    vector[Nsibs] mu_sib_phots;
    vector[Nsibs] mu_sib_phot_errs;

    vector[Ng] mu_ext_gal;
    vector[Ng] zcosmos;
    vector[Ng] zcosmoerrs;

    //real<lower=0,upper=1> sigma0;    //Data
    //real<lower=0,upper=1> pec_unity; //Data
}
transformed data{
  vector[Ng] mu_ext_gal_err_prefac;
  for (n in 1:Ng){
    mu_ext_gal_err_prefac[n] = 5/(log(10)*zcosmos[n]);
  }
}
parameters {
    vector[Ng] mu_true_gal;

    vector[Ng] eta_dM_common;
    vector[Nsibs] eta_dM_rel;

    real<lower=0,upper=1> sigma0;    //Model
    real<lower=0,upper=1> pec_unity; //Model
    real<lower=0,upper=1> eta_sigmaC_param;
}

transformed parameters {
    vector[Ng] dM_common;
    vector[Nsibs] dM_rel;
    vector[Nsibs] dM_sibs;
    real sigmaCommon;

    vector[Ng] mu_ext_gal_errs;
    mu_ext_gal_errs = mu_ext_gal_err_prefac .* sqrt(square(pec_unity)+square(zcosmoerrs));


    sigmaCommon = eta_sigmaC_param*sigma0;

    dM_common = eta_dM_common*sigmaCommon;
    dM_rel    = eta_dM_rel*sqrt(square(sigma0)-square(sigmaCommon));
    for (g in 1:Ng) {
      dM_sibs[sum(S_g[:g-1])+1:sum(S_g[:g])] = dM_rel[sum(S_g[:g-1])+1:sum(S_g[:g])] + dM_common[g];
    }
}

model {
    eta_dM_common ~ std_normal();
    eta_dM_rel    ~ std_normal();

    eta_sigmaC_param ~ uniform(0,1);
    sigma0    ~ uniform(0,1);
    pec_unity ~ uniform(0,1);

    for (g in 1:Ng){
      mu_sib_phots[sum(S_g[:g-1])+1:sum(S_g[:g])] ~ normal(mu_true_gal[g]+dM_sibs[sum(S_g[:g-1])+1:sum(S_g[:g])],mu_sib_phot_errs[sum(S_g[:g-1])+1:sum(S_g[:g])]);
    }

    mu_ext_gal ~ normal(mu_true_gal, mu_ext_gal_errs);
}

generated quantities {
  real<lower=0> sigmapec;
  real<lower=0> sigmaRel;
  real<lower=0,upper=1> rho;
  rho         = square(sigmaCommon)/square(sigma0);
  sigmaRel    = sqrt(square(sigma0)-square(sigmaCommon));
  sigmapec    = pec_unity*299792.458;
}
