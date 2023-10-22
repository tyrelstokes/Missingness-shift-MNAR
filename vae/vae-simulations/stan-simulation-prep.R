# source in functions -------

source(here::here("vae/vae-simulations/sim-full-data-funs.R"))
source(here::here("vae/results-organizing/organize-stan-results.R"))

# Working parameters -------



# Cmat gen function ----------------------------

cmat_gen <- function(n_pars,
                     n_child,
                     prob_1){
  
  Cmat <- matrix(rbinom(n_pars*n_child,size = 1,prob = prob_1), ncol = n_pars)
  Cmat
}


stan_sim_once <- function(n_z,
                          n_x,
                          n_r,
                          n,
                          mu_z,
                          sd_z,
                          sd_x,
                          sd_y,
                          alpha_vec_x,
                          is_missing,
                          r_prob_vec,
                          rand_cmat = FALSE,
                          prob_cmat = 0.7,
                          rand_beta = TRUE,
                          beta_sd = 1,
                          C_mat_x_z = NULL,
                          C_mat_x_x = NULL,
                          C_mat_r_x = NULL,
                          C_mat_r_r = NULL,
                          C_mat_y_x = NULL,
                          Beta_mat_x_z = NULL,
                          Beta_mat_x_x = NULL,
                          Beta_mat_r_x = NULL,
                          Beta_mat_r_r = NULL,
                          Beta_mat_y_x = NULL,
                          x_link_fun = "identity",
                          y_link_fun = "identity",                          n_nodes_x = c(2,2),
                          n_layers_x = 2,
                          n_nodes_y = c(2,2),
                          n_layers_y = 2,
                          act_fun_y = "relu",
                          final_act_y = "linear",
                          act_fun_x = "relu",
                          final_act_x = "linear"){
  
  if(rand_cmat ==TRUE){
    
    C_mat_x_z <- cmat_gen(n_pars = n_z,
                          n_child = n_x,
                          prob_1 = prob_cmat)
    
    C_mat_x_x <- cmat_gen(n_pars = n_x,
                          n_child = n_x,
                          prob_1 = prob_cmat)
    
    C_mat_r_x <- cmat_gen(n_pars = n_x,
                          n_child = n_r,
                          prob_1 = prob_cmat)
    
    C_mat_r_r <- cmat_gen(n_pars = n_r,
                          n_child = n_r,
                          prob_1 = prob_cmat)
    
    C_mat_y_x <- cmat_gen(n_pars = n_x,
                          n_child = 1,
                          prob_1 = prob_cmat)
    
    
  }
  
  
  if(rand_beta == TRUE){
   
    Beta_mat_x_z <- matrix(rnorm(n_z*n_x, sd = beta_sd),nrow = n_z)
    Beta_mat_x_x <- matrix(rnorm(n_x*n_x,sd = beta_sd),nrow = n_x)
    Beta_mat_r_x <- matrix(rnorm(n_r*n_x,sd = beta_sd),nrow = n_r)
    Beta_mat_r_r <- matrix(rnorm(n_r*n_r,sd = beta_sd),nrow = n_r)
    Beta_mat_y_x <- matrix(rnorm(1*n_r,sd = beta_sd),nrow = n_x) 
    
  }
  
  
  trial <-  sim_full_once(n = n,
                          n_z = n_z,
                          mu_z = mu_z,
                          sd_z = sd_z,
                          n_x = n_x,
                          C_mat_x_z = C_mat_x_z,
                          Beta_mat_x_z = Beta_mat_x_z,
                          sd_x = sd_x,
                          C_mat_x_x = C_mat_x_x,
                          Beta_mat_x_x = Beta_mat_x_x,
                          alpha_vec_x = alpha_vec_x,
                          n_r = n_r,
                          C_mat_r_x =C_mat_r_x,
                          Beta_mat_r_x = Beta_mat_r_x,
                          C_mat_r_r =  C_mat_r_r,
                          Beta_mat_r_r = Beta_mat_r_r,
                          is_missing = is_missing,
                          r_link_fun = "expit",
                          C_mat_y_x = C_mat_y_x,
                          Beta_mat_y_x = Beta_mat_y_x,
                          sd_y = sd_y,
                          r_prob_vec = r_prob_vec,
                          x_link_fun = x_link_fun,
                          y_link_fun = y_link_fun,
                          n_nodes_x = n_nodes_x,
                          n_layers_x = n_layers_x,
                          n_nodes_y = n_nodes_y,
                          n_layers_y = n_layers_x,
                          act_fun_y = act_fun_y,
                          final_act_y = final_act_y,
                          act_fun_x = act_fun_x,
                          final_act_x = final_act_x
  )
  
  
  
  alpha_u_true <- alpha_vec_x[2:3]
  psi_u_true <- Beta_mat_x_x[2,3]*C_mat_x_x[2,3]
  psi_o_true <- c(Beta_mat_x_x[1,2]*C_mat_x_x[1,2],
                  Beta_mat_x_x[1,3]*C_mat_x_x[1,3])
  
  full_data <- trial$full_data
  observed_data <- trial$obs_data
  x_lin_pred <- trial$x_lin_pred[,2:3]
  
  
  R <- cbind(observed_data$r_2,observed_data$r_3)
  x_u <- cbind(full_data$x_2,full_data$x_3)
  x_o <- full_data$x_1
  
  stan_list <- list(
    N = n,
    y = observed_data$y,
    R = R,
    x_u = x_u,
    x_o = x_o,
    alpha_u_true = alpha_u_true,
    psi_u_true = psi_u_true,
    psi_o_true = psi_o_true,
    x_lin_pred = x_lin_pred
    
  )
  
 full_data_lin_coefs_est <- trial$full_data_lin_coefs_est
 linear_coefs <- trial$linear_coefs

 emp_sd_vec_x <-  trial$emp_sd_vec_x
  
  out <- list(stan_list = stan_list,
              full_data = full_data,
              observed_data = observed_data,
              Beta_mat_x_z = Beta_mat_x_z,
              Beta_mat_x_x = Beta_mat_x_x,
              Beta_mat_r_x = Beta_mat_r_x,
              Beta_mat_r_r = Beta_mat_r_r,
              Beta_mat_y_x = Beta_mat_y_x,
              C_mat_x_z = C_mat_x_z,
              C_mat_x_x = C_mat_x_x,
              C_mat_r_x = C_mat_r_x,
              C_mat_r_r = C_mat_r_r,
              C_mat_y_x = C_mat_y_x,
              emp_sd_vec_x = emp_sd_vec_x,
              full_data_lin_coefs_est = full_data_lin_coefs_est,
              linear_coefs = linear_coefs)
  
  out
  
  
}





