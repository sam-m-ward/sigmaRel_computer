data {
    int<lower=0> Ns; // No. of siblings
    int<lower=1> Nf; // No. of features
    int<lower=0> Ng; // No. of siblings galaxies
    array[Ng] int<lower=2> S_g; //No. of siblings per galaxy

    matrix[Ns,Nf] X;    // Feature matrix
    matrix[Ns,Nf] Xerr; // Feature errors
    vector[Ns] Y;       // Predictor
    vector[Ns] Yerr;    // Predictor errors

    vector[Nf]   beta_prior;   // Prior width on slope parameter
    vector[Nf+1] alpha_prior;  // Prior width on alpha parameters        (index 1 is y-disp)
    vector[Nf+1] sigint_prior; // Prior width on y and x-disp parameters (index 1 is y-disp)
    vector[Nf+1] alpha_const;  // Added constant on alpha parameters     (index 1 is y-disp)
    vector[Nf+1] sigint_const; // Added constant on sigint parameters    (index 1 is y-disp)
}

parameters {
    vector<lower=0,upper=1>[Nf] etabeta;
    vector[Ns*Nf] eta_x_rel;
    vector[Ng*Nf] eta_x_common;
    vector[Ng] eta_y_common;                  // transformed common components of latent predictor parameters
    vector<lower=0,upper=1>[Nf+1] etaalpha;
    vector<lower=0,upper=1>[Nf+1] etasigint;
    vector<lower=0,upper=1>[Nf+1] rho;
    vector[Ns] y;                             // latent predictor parameters
}

transformed parameters {
    vector[Nf] beta;      // slope hyperparameters
    vector[Ng] y_common;
    matrix[Ns,Nf] x_rel;
    matrix[Ng,Nf] x_common;
    matrix[Ns,Nf] x;
    vector[Nf+1] alpha;
    vector[Nf+1] sigmaint;
    vector[Nf+1] sigmaRel;
    vector[Nf+1] sigmaCommon;

    beta        = beta_prior   .* tan( pi() * (etabeta -0.5) );
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
    y_common = eta_y_common * sigmaCommon[1];

    for (g in 1:Ng) {
      x[sum(S_g[:g-1])+1:sum(S_g[:g]),:] = x_rel[sum(S_g[:g-1])+1:sum(S_g[:g]),:];
      for (f in 1:Nf) {
        x[sum(S_g[:g-1])+1:sum(S_g[:g]),f] += x_common[g,f]+alpha[1+f];
      }
    }
}

model {
    etabeta       ~ uniform(0,1);
    eta_x_rel     ~ std_normal();
    eta_x_common  ~ std_normal();
    eta_y_common  ~ std_normal();
    etaalpha      ~ uniform(0,1);
    etasigint     ~ uniform(0,1);
    rho           ~ beta(0.5,0.5);

    for (g in 1:Ng){
      y[sum(S_g[:g-1])+1:sum(S_g[:g])] ~ normal(x[sum(S_g[:g-1])+1:sum(S_g[:g]),:]*beta + y_common[g] + alpha[1], sigmaRel[1]); // Model
    }

    for (f in 1:Nf) {
      X[:,f] ~ normal(x[:,f],Xerr[:,f]); // Measurement-likelihood of features
    }
    Y ~ normal(y, Yerr);                 // Measurement-likelihood of predictor
}

generated quantities {
  vector[Ns] res;
  for (g in 1:Ng) {
    res[sum(S_g[:g-1])+1:sum(S_g[:g])] = y[sum(S_g[:g-1])+1:sum(S_g[:g])] - x[sum(S_g[:g-1])+1:sum(S_g[:g]),:]*beta - y_common[g]-alpha[1];  // Residuals for each data point
  }
}
