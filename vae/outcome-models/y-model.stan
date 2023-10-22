//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  vector[N] y;
  matrix[N,2] x_u;
  vector[N] x_o;
  matrix[N,2] R;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real alpha;
  real beta_o;
  vector[2] beta_u;
  real<lower=0> sigma;
  
  real alpha_cc;
  real beta_o_cc;
  vector[2] beta_u_cc;
  real <lower = 0> sigma_cc;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  y ~ normal(alpha + x_o*beta_o + x_u*beta_u, sigma);
  
  alpha ~ normal(0,5);
  beta_o ~ normal(0,5);
  beta_u ~ normal(0,5);
  sigma ~ normal(0,3);
  
  for(i in 1:N){
    if(sum(R[i,:]) < 2){
      
    }else{
      y[i] ~ normal(alpha_cc + x_o[i]*beta_o_cc + x_u[i,:]*beta_u_cc,sigma_cc);
    }
  }
  
  alpha_cc ~ normal(0,5);
  beta_o_cc ~ normal(0,5);
  beta_u_cc ~ normal(0,5);
  sigma_cc ~ normal(0,3);
  
}generated quantities{
  
  vector[N] pred;
  vector[N] pred_cc;
  
  for(i in 1:N){
    pred[i] = alpha + x_o[i]*beta_o + x_u[i,:]*beta_u;
    pred_cc[i] = alpha_cc + x_o[i]*beta_o_cc + x_u[i,:]*beta_u_cc;
    
  }
  
}

