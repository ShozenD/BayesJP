functions {
  matrix break_point(int N, int J, vector t, array[] real tau){
    matrix[N, J] A, B;
    matrix[N, J] P1, P2;

    vector[J] c1, c2, d1, d2;
    vector[J] gamma1, gamma2;
    matrix[N, J] BP;            // Breakpoint features

    for (j in 1:J){
      for (i in 1:N){
        A[i,j] = t[i] - tau[j];
        B[i,j] = t[i] * A[i,j];

        P1[i,j] = step(-A[i,j]);
        P2[i,j] = 1 - P1[i,j];
      }

      c1[j] = A[:,j]' * P1[:,j];
      c2[j] = A[:,j]' * P2[:,j];
      d1[j] = B[:,j]' * P1[:,j];
      d2[j] = B[:,j]' * P2[:,j];

      gamma1[j] = (N*(N+1)/2.0 * c2[j] - N*d2[j]) / (d2[j]*c1[j]-d1[j]*c2[j]);
      gamma2[j] = (d1[j]*N - N*(N+1)/2.0 * c1[j]) / (d2[j]*c1[j]-d1[j]*c2[j]);

      for(i in 1:N){
        BP[i,j] = 1 + (t[i]-tau[j]) * gamma1[j] + (gamma2[j] - gamma1[j]) * step(t[i]-tau[j]);
      }
    }

    return(BP);
  }
}

data {
  int<lower=0> N;  // No. of obs.
  array[N] int Y;  // The no. of deaths at each time point
  vector[N] t;     // Timepoints
  int J;           // No. of join points (TODO: make adjustable)
  int J2;          // 2^J
  matrix[J, J2] Delta; // Matrix with all combinations of delta
  array[J2] int IDX_NCP;
  vector[N] P;     // Population size at each time point
}

transformed data {
  real tmin = min(t);
  real tmax = max(t);
  real tmean = (tmax - tmin) / 2;

  real tauinf = tmin + 2;
  real tausup = tmax - 2;
}

parameters {
  real alpha;
  real beta0;
  array[J] real<lower=tauinf, upper=tausup> tau;   // Change points
  vector[J] beta_j;                     // Break point effects
  real<lower=0> gamma;
  vector[J] p;
}

transformed parameters {
  vector[J2] lp = rep_vector(0, J2); // Vector to house the joint-log posterior density

  { // Marginalise out delta
    // Calculate the precision matrix
    matrix[N, J] B = break_point(N, J, t, tau);
    // vector[N] w = exp(alpha + beta0 * (t - tmean) + log(P));
    vector[N] w = sqrt(exp(alpha + beta0 * (t - tmean) + log(P)));

    matrix[J, J] BWB = quad_form_sym(diag_matrix(w), B); // B'WB

    for (k in 1:J2){
      vector[J] delta = Delta[:,k];
      matrix[J, J] DBWBD = quad_form_sym(BWB, diag_matrix(delta));
      matrix[J, J] Sigma = N*inverse(add_diag(DBWBD, diagonal(BWB - DBWBD)));

      // Marginalised likelihood
      lp[k] = lp[k] + multi_normal_lpdf(beta_j | rep_vector(0, J),  gamma*Sigma) + log(J^(-J)*(J-1)^(J - sum(delta)));
      vector[N] log_mu = alpha + beta0 * (t - tmean) + B * (beta_j .* delta) + log(P);
      lp[k] = lp[k] + poisson_lpmf(Y | exp(log_mu));
    }
  }
}

model {
  // Priors for common parameters (uninformative)
  target += normal_lpdf(alpha | 0, 10);
  target += normal_lpdf(beta0 | 0, 10);

  { // local scope
    // The following trick ensures that tau is ordered
    // real condition = step(tau[2]-tau[1]-2)*step(tau[3]-tau[2]-2)*step(tau[4]-tau[3]-2);
    real condition = step(tau[2]-tau[1]-2);
    target += bernoulli_lpmf(1 | condition);
  }

  target += uniform_lpdf(tau | tauinf, tausup); // Uniform prior on tau
  target += inv_gamma_lpdf(gamma | 1, 0.5);

  target += log_sum_exp(lp); // Add in the marginalised likelihoo
}

generated quantities {
   vector[N] mu;    // Estimated rates
   array[N] int yhat;  // Predicted deaths
   array[N-1] real APC; // Average percentage change

   {// Local scope
    // Calculate precision matrix
    vector[J] delta = rep_vector(1, J);

    matrix[N, J] B = break_point(N, J, t, tau);
    vector[N] w = sqrt(P .* exp(alpha + beta0 * (t - tmean)));
    matrix[J, J] BWB = quad_form_sym(diag_matrix(w), B);
    matrix[J, J] DBWBD =  diag_pre_multiply(delta, diag_post_multiply(BWB, delta));
    matrix[J, J] Sigma = inverse_spd(add_diag(DBWBD, diagonal(BWB - DBWBD)));

    mu = exp(alpha + beta0 * (t - tmean) + B * beta_j + log(P));
    yhat = poisson_rng(mu);

    for(i in 1:(N-1)){
      APC[i] = 100 * (mu[i+1] - mu[i]) / mu[i];
    }
   }
}
