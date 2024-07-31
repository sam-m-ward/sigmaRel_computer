functions {
  real usamp_sp(real rho, real p) {
  		real p_rho;
      p_rho = sqrt(pow(rho,p-2)+pow(1-rho,p-2));
  		return log(p_rho);
  }
  real usamp_thetap(real rho, real p) {
  		real p_rho;
      real A;
      real B;
      real C;

      A = 1/(sqrt(rho*(1-rho)));
      B = pow(rho,(1+p)/2)*pow(1-rho,(1-p)/2);
      C = pow(rho,(1-p)/2)*pow(1-rho,(1+p)/2);
      p_rho = A/(B+C);

  		return log(p_rho);
  }
  real usamp_plaw(real rho, real p) {
      real p_rho;
      p_rho = pow(sqrt(rho*(1-rho)),p-2);
      return log(p_rho);
  }
}

data {
    int<lower=0> Ng;
    array[Ng] int<lower=2> S_g;
    int <lower=Ng*2> Nsibs;

    vector[Nsibs] mu_sib_phots;
    vector[Nsibs] mu_sib_phot_errs;

    vector[Ng] mu_ext_gal;
    vector[Ng] zcosmos;
    vector[Ng] zcosmoerrs;

    real p;
    real<lower=0,upper=2> usamp_input; //define 0 as uniform on sp; 1 as uniform on thetap

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

    real<lower=0,upper=1> sigma0;     //Model
    real<lower=0,upper=1> rho;        //Model
    real<lower=0,upper=1> pec_unity;  //Model
}

transformed parameters {
    vector[Ng] dM_common;
    vector[Nsibs] dM_rel;
    vector[Nsibs] dM_sibs;

    real sigmaRel;      //Model
    real sigmaCommon;   //Model

    sigmaCommon = sqrt(rho)*sigma0;
    sigmaRel    = sqrt(square(sigma0)-square(sigmaCommon));

    vector[Ng] mu_ext_gal_errs;
    mu_ext_gal_errs = mu_ext_gal_err_prefac .* sqrt(square(pec_unity)+square(zcosmoerrs));

    dM_rel    = eta_dM_rel*sigmaRel;
    dM_common = eta_dM_common*sigmaCommon;
    for (g in 1:Ng) {
      dM_sibs[sum(S_g[:g-1])+1:sum(S_g[:g])] = dM_rel[sum(S_g[:g-1])+1:sum(S_g[:g])] + dM_common[g];
    }
}

model {
    eta_dM_common ~ std_normal();
    eta_dM_rel    ~ std_normal();

    sigma0  ~ uniform(0,1);
    if (usamp_input==0) {
      target += usamp_sp(rho,p);
    } else if (usamp_input==1) {
      target += usamp_thetap(rho,p);
    } else if (usamp_input==2) {
      target += usamp_plaw(rho,p);
    }
    pec_unity   ~ uniform(0,1);

    for (g in 1:Ng){
      mu_sib_phots[sum(S_g[:g-1])+1:sum(S_g[:g])] ~ normal(mu_true_gal[g]+dM_sibs[sum(S_g[:g-1])+1:sum(S_g[:g])],mu_sib_phot_errs[sum(S_g[:g-1])+1:sum(S_g[:g])]);
    }

    mu_ext_gal ~ normal(mu_true_gal, mu_ext_gal_errs);
}

generated quantities {
  real<lower=0> sigmapec;
  sigmapec = pec_unity*299792.458;
}
