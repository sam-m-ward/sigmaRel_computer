data {
    int<lower=0> Nd; // No. of data points
    int<lower=1> Nf; // No. of features

    matrix[Nd,Nf] X;    // Feature matrix
    matrix[Nd,Nf] Xerr; // Feature errors
    vector[Nd] Y;       // Predictor
    vector[Nd] Yerr;    // Predictor errors

    real alpha_prior;   // Prior width on const parameter
    real beta_prior;    // Prior width on const parameter
    real sigint_prior;  // Prior width on const parameter

    vector[Nf] beta;     // Data slope hyperparameters
}

parameters {
    matrix[Nd,Nf] x;      // latent feature parameters
    vector[Nd] y;         // Latent predictor parameters
    real etaalpha;           // Constant
    real<lower=0> etasigint; // intrinsic scatter in multiple linear regression
    //vector[Nf] etabeta;      // Model slope hyperparameters
}

transformed parameters {
    real alpha;           // Constant
    //vector[Nf] beta;    // Model slope hyperparameters
    real<lower=0> sigint; // intrinsic scatter in multiple linear regression

    alpha  = alpha_prior*etaalpha;
    //beta   = beta_prior*etabeta;
    sigint = sigint_prior*etasigint;
}

model {
    etaalpha  ~ std_normal();
    etasigint ~ std_normal();
    //etabeta   ~ std_normal();

    for (f in 1:Nf) {
      X[:,f] ~ normal(x[:,f],Xerr[:,f]);        // Measurement-likelihood of features
    }
    y ~ normal(x*beta + alpha, sigint); // Model
    Y ~ normal(y, Yerr);                // Measurement-likelihood of predictor
}
generated quantities {
  vector[Nd] res;
  real sigmaRel;
  sigmaRel = sigint*sqrt(2);
  res = y - x*beta - alpha;  // Residuals for each data point
}
