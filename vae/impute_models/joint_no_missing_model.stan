
data {
  int<lower=0> N;
  vector[N] y;
  array[N,2] int R;
  matrix[N,2] x_u;
  vector[N] x_o;
  vector[2] psi_o_true;
  vector[2] alpha_u_true;
  real psi_u_true;
  matrix[N,2] x_lin_pred;
}
transformed data{
  
  array[2] int n_miss;
  array[2] int n_obs;
  
  n_obs[1] = sum(R[:,1]);
  n_obs[2] = sum(R[:,2]);
  
  n_miss[1] = N - n_obs[1];
  n_miss[2] = N - n_obs[2];
  
  
  
}

parameters {
  vector[2] alpha_u;
  vector[2] psi_o;
  vector<lower =0>[2] sigma_u;
 // vector[2] alpha_r;
  //vector[2] gamma_u;
  //vector[2] gamma_o;
 // real gamma_u1;
  //real alpha_y;
  //vector[2] beta_u;
  //real beta_o;
 // real<lower=0> sigma_y;
  real psi_u;
  vector[n_miss[1]] x_u_1_miss_0;
  vector[n_miss[2]] x_u_2_miss_0;
}

model {

vector[N] x_u_re_1;
vector[N] x_u_re_2;
vector[n_miss[1]] x_u_miss_1;
vector[n_miss[2]] x_u_miss_2;

vector[n_obs[1]] x_u_obs_1;
vector[n_obs[2]] x_u_obs_2;

vector[n_obs[1]] x_o_obs_1;
vector[n_obs[2]] x_o_obs_2;

vector[n_obs[2]] x_u_covs;



//vector[N] mu_r1;
//vector[N] mu_r2;



int j;
int k;
int l;
int m;

j = 1;
k = 1;

l = 1;
m = 1;

for(i in 1:N){
  
  if(R[i,1] == 1){
    x_u_re_1[i] = x_u[i,1];
    //x_u[i,1] ~ normal(alpha_u + x_o_obs[i]*psi_o[1],sigma_u[i])
    x_u_obs_1[l] = x_u[i,1];
    x_o_obs_1[l] = x_o[i];
    l += 1;
     //normal(alpha_u + x_o_obs*psi_o,sigma_u);
  }else{
  
  
  x_u_re_1[i] = alpha_u[1] + x_o[i]*psi_o[1] + x_u_1_miss_0[j]*sigma_u[1];
  
  x_u_miss_1[j] = x_u_re_1[i];
  j += 1;
  
}

if(R[i,2] == 1){
   x_u_re_2[i] = x_u[i,2];
    //x_u[i,1] ~ normal(alpha_u + x_o_obs[i]*psi_o[1],sigma_u[i])
    x_u_obs_2[m] = x_u[i,2];
    x_o_obs_2[m] = x_o[i];
    x_u_covs[m] = x_u_re_1[i];
    m += 1;
  
}else{
  
  x_u_re_2[i] = alpha_u[2] + x_o[i]*psi_o[2] + x_u_re_1[i]*psi_u+x_u_2_miss_0[k]*sigma_u[2];
  
  x_u_miss_2[k] = x_u_re_2[i];
  k += 1;
  
  
}

}

x_u_obs_1 ~ normal(alpha_u[1] + x_o_obs_1*psi_o[1],sigma_u[1]);
x_u_obs_2 ~ normal(alpha_u[2] + x_o_obs_2*psi_o[2] + x_u_covs*psi_u, sigma_u[2]);

//mu_r1 = alpha_r[1] + x_o*gamma_o[1] + x_u_re_1*gamma_u[1];
//mu_r2 = alpha_r[2] + x_o*gamma_o[2] + x_u_re_2*gamma_u[2] + x_u_re_1*gamma_u1;

//R[:,1] ~ bernoulli_logit(mu_r1);
//R[:,2] ~ bernoulli_logit(mu_r2);


 //y ~ normal(alpha_y + x_u_re_1*beta_u[1] +x_u_re_2*beta_u[2] + x_o*beta_o, sigma_y);
 
 alpha_u ~ normal(0,2);
 //alpha_y ~ normal(0,2);
 //alpha_r ~ normal(0,2);
 //gamma_u ~ normal(0,2);
 //gamma_u1 ~ normal(0,2);
 //gamma_o ~ normal(0,2);
 psi_o ~ normal(0,2);
 psi_u ~ normal(0,2);
 
 //sigma_y ~ normal(0,5);
 sigma_u ~ normal(0,1);
 
 //beta_u ~ normal(0,5);
 //beta_o ~ normal(0,5);
 
 x_u_1_miss_0 ~ normal(0.0,1.0);
 x_u_2_miss_0 ~ normal(0.0,1.0);






}generated quantities{

vector[n_miss[1]] x_u_true_1;
vector[n_miss[2]] x_u_true_2;

vector[n_miss[1]] x_u_miss_1;
vector[n_miss[2]] x_u_miss_2;

vector[n_miss[1]] exp_x_u_1;
vector[n_miss[2]] exp_x_u_2;

vector[N] x_u_re_1;
vector[N] x_u_re_2;


real x_u_int;
int j;
int k;
int l;
int m;

j = 1;
m = 1;


for(i in 1:N){
  
  if(R[i,1] == 0){
    
  x_u_true_1[j] = x_u[i,1];
  x_u_miss_1[j] = alpha_u[1] + x_o[i]*psi_o[1] + x_u_1_miss_0[j]*sigma_u[1];
  x_u_re_1[i] = x_u_miss_1[j];
  exp_x_u_1[j] = x_lin_pred[i,1];
  j += 1;
  
}else{
  x_u_re_1[i] = x_u[i,1];
  
}

if(R[i,2] == 0){
  x_u_true_2[m] = x_u[i,2];
  exp_x_u_2[m] = x_lin_pred[i,2];
  
  if(R[i,1] == 1){
  x_u_miss_2[m] = alpha_u[2] + x_o[i]*psi_o[2] + x_u[i,1]*psi_u+x_u_2_miss_0[m]*sigma_u[2];
  x_u_re_2[i] = x_u_miss_2[m];
  }else{
    x_u_int = alpha_u[1] + x_o[i]*psi_o[1] + x_u_1_miss_0[(j-1)]*sigma_u[1];
    x_u_miss_2[m] = alpha_u[2] + x_o[i]*psi_o[2] + x_u_int*psi_u+x_u_2_miss_0[m]*sigma_u[2];
    x_u_re_2[i] = x_u_miss_2[m];
  }
    m += 1;
  
}else{
  x_u_re_2[i] = x_u[i,2];
}

}
}



