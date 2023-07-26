data {
  int<lower=1> S;       //Total No. of Siblings
  vector[S] mu_s;       //Vector of mu estimates
  vector[S] mu_err_s;   //Vector of mu estimate errors (fitting uncertainties)

  real mean_mu;         //Initial guess of mu used for uninformative prior
  real<lower=0> sigma0; //maximum upper bound on sigma0==sigmaRel
}

parameters{
  vector[S] eta_dM_Rel_s; //Indep. drawn dMs
  real mu;                //Common mu
}

transformed parameters{
  vector[S] dM_Rel_s;

  dM_Rel_s  = sigma0    * eta_dM_Rel_s;

}

model{
  eta_dM_Rel_s  ~ std_normal();
  mu            ~ normal(mean_mu,100);

  mu_s          ~ normal(mu+dM_Rel_s, mu_err_s);
}
