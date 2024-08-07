data {
    int<lower=0> Ns; // No. of siblings
    int<lower=1> Nf; // No. of features
    int<lower=0> Ng; // No. of siblings galaxies
    array[Ng] int<lower=2> S_g; //No. of siblings per galaxy

    matrix[Ns,Nf] X;    // Feature matrix
    matrix[Ns,Nf] Xerr; // Feature errors
    vector[Ns] Y;       // Predictor
    vector[Ns] Yerr;    // Predictor errors

    real alpha_prior;   // Prior width on const parameter
    real beta_prior;    // Prior width on const parameter
    real sigRel_prior;  // Prior width on const parameter

    vector[Nf] beta;     // Data slope hyperparameters
}

parameters {
    matrix[Ns,Nf] x;  // latent feature parameters
    vector[Ns] y;     // latent predictor parameters
    vector[Ng] mu_g;  // latent galaxy distances
    real<lower=0,upper=1>       etaalpha;     // Constant
    real<lower=0,upper=1>       etasigRel;    // intrinsic scatter in multiple linear regression
}

transformed parameters {
    real alpha;           // Constant
    real<lower=0> sigmaRel; // intrinsic scatter in multiple linear regression

    alpha    = alpha_prior  * tan( pi()    * (etaalpha-0.5));
    sigmaRel = sigRel_prior * etasigRel;
}

model {
    etaalpha  ~ uniform(0,1);
    etasigRel ~ uniform(0,1);

    for (g in 1:Ng){
      y[sum(S_g[:g-1])+1:sum(S_g[:g])] ~ normal(x[sum(S_g[:g-1])+1:sum(S_g[:g]),:]*beta + mu_g[g] + alpha, sigmaRel); // Model
    }
    for (f in 1:Nf) {
      X[:,f] ~ normal(x[:,f],Xerr[:,f]); // Measurement-likelihood of features
    }
    Y ~ normal(y, Yerr);                 // Measurement-likelihood of predictor
}

generated quantities {
  vector[Ns] res;
  for (g in 1:Ng) {
    res = y - x*beta - mu_g[g] - alpha;  // Residuals for each data point
  }
}
