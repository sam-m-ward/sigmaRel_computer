data {
  int<lower=1> S;       //Total No. of Siblings
  vector[S] mu_s;       //Vector of mu estimates
  vector[S] mu_err_s;   //Vector of mu estimate errors (fitting uncertainties)

  real mean_mu;         //Initial guess of mu used for uninformative prior
  real<lower=0> sigma0; //maximum upper bound on sigma0==sigmaRel

  real<lower=0,upper=1> asymmetric; //SigmaRel Hyperprior
}

parameters{
  //real<lower=0, upper=sigma0> sigmaRel; //Relative Scatter
  real<lower=0,upper=1> rho;            //Correlation
  vector[S] eta_dM_Rel_s;               //Indep. drawn dMs
  real eta_dM_Common;                   //Common dM
  real mu;                              //Common mu
}

transformed parameters{
  real<lower=0> sigmaRel;
  real<lower=0> sigmaCommon;
  vector[S] dM_Rel_s;
  real dM_Common;

  sigmaRel    = sqrt(1-rho)*sigma0;
  sigmaCommon = sqrt(rho)*sigma0;     //sqrt(square(sigma0)-square(sigmaRel));

  dM_Rel_s  = sigmaRel    * eta_dM_Rel_s;
  dM_Common = sigmaCommon * eta_dM_Common;
}

model{
  if (asymmetric==0) {
    rho ~ beta(0.5,0.5);
  } else if (asymmetric==1) {
    target += -log(sqrt(1-rho)); //equiv to sigR~ uniform(0,sig0)
  }
  eta_dM_Rel_s  ~ std_normal();
  eta_dM_Common ~ std_normal();
  mu            ~ normal(mean_mu,100);

  mu_s          ~ normal(mu+dM_Rel_s+dM_Common, mu_err_s);
}
