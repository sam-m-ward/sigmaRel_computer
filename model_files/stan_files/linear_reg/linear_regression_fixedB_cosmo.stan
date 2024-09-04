data {
    int<lower=0> Ns; // No. of siblings
    int<lower=1> Nf; // No. of features
    int<lower=0> Ng; // No. of siblings galaxies
    array[Ng] int<lower=2> S_g; //No. of siblings per galaxy
    int<lower=1> Nb; // No. of beta hyperparameters

    matrix[Ns,Nf] X;    // Feature matrix
    matrix[Ns,Nf] Xerr; // Feature errors
    vector[Ns] Y;       // Predictor
    vector[Ns] Yerr;    // Predictor errors

    vector[Nb]   beta;  // Data slope hyperparameters

    vector[Nf+1] alpha_prior;  // Prior width on alpha parameters        (index 1 is y-disp)
    vector[Nf+1] sigint_prior; // Prior width on y and x-disp parameters (index 1 is y-disp)
    vector[Nf+1] alpha_const;  // Added constant on alpha parameters     (index 1 is y-disp)
    vector[Nf+1] sigint_const; // Added constant on sigint parameters    (index 1 is y-disp)

    vector[Ng] mu_ext_gal;
    vector[Ng] zcosmos;
    vector[Ng] zcosmoerrs;

    int<lower=0,upper=1> zero_beta_common; // if zero, no common component of features included, if one, include common component

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
    vector[Ns*Nf] eta_x_rel;
    vector[Ng*Nf] eta_x_common;
    vector[Ns] eta_y_rel;                     // transformed relative components of latent predictor parameters
    vector[Ng] eta_y_common;                  // transformed common components of latent predictor parameters
    vector<lower=0,upper=1>[Nf+1] etaalpha;
    vector<lower=0,upper=1>[Nf+1] etasigint;
    vector<lower=0,upper=1>[Nf+1] rho;

    vector[Ng] mu_true_gal;
    real<lower=0,upper=1> pec_unity;  //Model
}

transformed parameters {
    vector[Ns] y_rel;
    vector[Ng] y_common;
    matrix[Ns,Nf] x_rel;
    matrix[Ng,Nf] x_common;
    matrix[Ns,Nf] x;      // X
    vector[Ns] xb;        // X@beta
    vector[Nf+1] alpha;
    vector[Nf+1] sigmaint;
    vector[Nf+1] sigmaRel;
    vector[Nf+1] sigmaCommon;
    vector[Ng] mu_ext_gal_errs;

    mu_ext_gal_errs = mu_ext_gal_err_prefac .* sqrt(square(pec_unity)+square(zcosmoerrs));

    alpha       = alpha_prior  .* tan( pi() * (etaalpha-0.5) ) + alpha_const;
    sigmaint    = sigint_prior .* etasigint + sigint_const;
    sigmaRel    = sqrt(1-rho)  .* sigmaint;
    sigmaCommon = sqrt(rho)    .* sigmaint;

    x_rel    = to_matrix(eta_x_rel,Ns,Nf);
    x_common = to_matrix(eta_x_common,Ng,Nf);
    for (f in 1:Nf){
      x_rel[:,f]    = x_rel[:,f]    * sigmaRel[1+f];
      x_common[:,f] = x_common[:,f] * sigmaCommon[1+f];
    }
    y_rel    = eta_y_rel    * sigmaRel[1];
    y_common = eta_y_common * sigmaCommon[1];

    for (g in 1:Ng) {
      x[sum(S_g[:g-1])+1:sum(S_g[:g]),:] = x_rel[sum(S_g[:g-1])+1:sum(S_g[:g]),:];
      for (f in 1:Nf) {
        x[sum(S_g[:g-1])+1:sum(S_g[:g]),f]  += x_common[g,f]+alpha[1+f];
        xb[sum(S_g[:g-1])+1:sum(S_g[:g])]  = beta[f] * x_rel[sum(S_g[:g-1])+1:sum(S_g[:g]),f];
        xb[sum(S_g[:g-1])+1:sum(S_g[:g])] += beta[Nb-Nf+f]*x_common[g,f]*zero_beta_common+alpha[1+f];
      }
    }
}

model {
    eta_x_rel     ~ std_normal();
    eta_x_common  ~ std_normal();
    eta_y_rel     ~ std_normal();
    eta_y_common  ~ std_normal();
    etaalpha      ~ uniform(0,1);
    etasigint     ~ uniform(0,1);
    rho           ~ beta(0.5,0.5);
    pec_unity     ~ uniform(0,1);

    for (g in 1:Ng){
      Y[sum(S_g[:g-1])+1:sum(S_g[:g])] ~ normal(xb[sum(S_g[:g-1])+1:sum(S_g[:g])] + y_rel[sum(S_g[:g-1])+1:sum(S_g[:g])] + y_common[g] + alpha[1] + mu_true_gal[g], Yerr[sum(S_g[:g-1])+1:sum(S_g[:g])]); // Measurement-likelihood of photometric distances
    }
    mu_ext_gal ~ normal(mu_true_gal, mu_ext_gal_errs); // Measurement-likelihood of cosmo distances

    for (f in 1:Nf) {
      X[:,f] ~ normal(x[:,f],Xerr[:,f]); // Measurement-likelihood of features
    }
}

generated quantities {
  real<lower=0> sigmapec;

  sigmapec = pec_unity*299792.458;
}
