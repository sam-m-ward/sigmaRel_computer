data {
  int<lower=1> S;       //Total No. of Siblings
  vector[S] mu_s;       //Vector of mu estimates
  vector[S] mu_err_s;   //Vector of mu estimate errors (fitting uncertainties)

  real mean_mu;         //Initial guess of mu used for uninformative prior
  real<lower=0> sigma0; //maximum upper bound on sigma0==sigmaRel
}

parameters{
  real eta_dM_Common;     //Common dM
  real mu;                //Common mu
}

transformed parameters{
  real dM_Common;

  dM_Common = sigma0 * eta_dM_Common;
}

model{
  eta_dM_Common ~ std_normal();
  mu            ~ normal(mean_mu,100);

  mu_s          ~ normal(mu+dM_Common, mu_err_s);
}
