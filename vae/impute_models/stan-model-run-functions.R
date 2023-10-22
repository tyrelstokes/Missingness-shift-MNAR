# Housekeeping -----------------

source(here::here("vae/results-organizing/organize-stan-results.R"))
source(here::here("vae/vae-simulations/sim-full-data-funs.R"))
source(here::here("vae/vae-simulations/stan-simulation-prep.R"))
source(here::here("vae/outcome-models/run-outcome-models.R"))
`%>%` <- dplyr::`%>%`
`%do%` <- foreach::`%do%`

# imputed data set function ------------

imputed_df <- function(stan_list,
                       x_u_1_fit,
                       x_u_2_fit){
  
  R <- stan_list$R

  x_u <- stan_list$x_u
  
  x_u_imputed <- x_u
  
  x_u_imputed[R[,1]==0,1] <- x_u_1_fit
  x_u_imputed[R[,2]==0,2] <- x_u_2_fit
  
  x_u_imputed
  
}


# run model function ---------------


run_joint <- function(stan_list,
                      model_file,
                      mod_name,
                      init,
                      iter,
                      fit_method,
                      n_chains = 4,
                      n_par_chains = 4,
                      iter_warmup = 1000,
                      iter_sampling = 1000,
                      index = 1,
                      use_rstan =FALSE){
  
  
  if(use_rstan == FALSE){
  impute_mod <- cmdstanr::cmdstan_model(
    model_file
  )
  
  if(fit_method == 'vb'){
    
    
    fit <- impute_mod$variational(data = stan_list,
                                        init = init,
                                        iter = iter)
  }
  
if(fit_method =="mcmc"){
  fit <- impute_mod$sample(data = stan_list,
                           init = init,
                           chains = n_chains,
                           parallel_chains = n_par_chains,
                           iter_warmup = iter_warmup,
                           iter_sampling = iter_sampling)
}  
  

  if(fit_method == "opt"){
    fit <- impute_mod$sample(data = stan_list,
                             init = init)
  }  
  
  }else{
    
    impute_mod <- rstan::stan_model(
      model_file
    )
    
    if(fit_method == 'vb'){
      
      fit <- rstan::vb(object = impute_mod,
                       data = stan_list,
                       init = init,
                       iter = iter)
      
    }
    
    if(fit_method =='mcmc'){
      
      fit <- rstan::sampling(object = impute_mod,
                             chains = n_chains,
                             iter = (iter_warmup + iter_sampling),
                             init = init,
                             cores = n_par_chains)
      
      
    }
    
    if(fit_method =="opt"){
      fit <- rstan::optimizing(object = impute_mod,
                               hessian = TRUE,
                               importance_resampling = TRUE,
                               draws = 1000)
    }
    
  }
  
  if(mod_name %in% c("Z model", "Joint Model")){
    outcome_pred <- FALSE
  }else{
    outcome_pred <- TRUE
  }
  
 model_objects <-  extract_function(fitted_model = fit,
                                    mod_name = mod_name,
                                    fit_type = fit_method,
                                    index = index,
                                    use_rstan = use_rstan,
                                    outcome_pred = outcome_pred,
                                    y = stan_list$y,
                                    x_o = stan_list$x_o,
                                    R = stan_list$R)
 

 y_model_df <- model_objects$y_model_df 
 
 combined_df <- model_objects$comb
 rmse_df <- model_objects$rmse_df
 model_summary <- model_objects$mod_sum
 
 x_u_1_fit = model_objects$x_u_1_fit
 x_u_2_fit = model_objects$x_u_2_fit
 
 y_preds <- model_objects$y_preds
 
 x_u_imputed <- imputed_df(stan_list = stan_list,
                           x_u_1_fit = x_u_1_fit,
                           x_u_2_fit = x_u_2_fit)
 
 

 
 out <- list(rmse_df = rmse_df,
             combined_df = combined_df,
             model_summary = model_summary,
             fitted_model = fit,
             x_u_1_fit = x_u_1_fit,
             x_u_2_fit = x_u_2_fit,
             x_u_imputed = x_u_imputed,
             y_preds = y_preds,
             y_model_df = y_model_df)
  
}

# Name function ----------------

mod_namer <- function(mod_file){
  if(mod_file == here::here("vae/impute_models/joint_impute_model_3.stan")){
    out <- "Joint Model"
  }
  if(mod_file == here::here("vae/impute_models/joint_no_outcome.stan")){
    out <- "No outcome model"
  }
  
  if(mod_file == here::here("vae/impute_models/joint_no_missing_model.stan")){
    out <- "No missingness model"
  }
  
  if(mod_file ==here::here("vae/impute_models/z_model.stan")){
    out <- "Z model"
  }

out

}


mod_retriever <- function(mod_name){
  if(mod_name == "Joint Model"){
    out <- here::here("vae/impute_models/joint_impute_model_3.stan")
  }
  
  if(mod_name == "No outcome model"){
    out <- here::here("vae/impute_models/joint_no_outcome.stan")
  }
  
  if(mod_name == "No missingness model"){
    out <- here::here("vae/impute_models/joint_no_missing_model.stan")
  }
  
  if(mod_name == "Z model"){
    out <- here::here("vae/impute_models/z_model.stan")
  }
  
  out
  
}
# Several Methods function --------------------

run_joint_many <- function(stan_list,
                      model_name_vec,
                      init,
                      iter,
                      fit_method_vec,
                      n_chains = 4,
                      n_par_chains = 4,
                      iter_warmup = 1000,
                      iter_sampling = 1000,
                      index = 1,
                      use_rstan = FALSE){
  
  n_mods <- length(model_name_vec)
  n_methods <- length(fit_method_vec)
  
  impute_list <- vector('list',length = n_mods)
  y_models <- vector('list', length = n_mods)
  
  model_file_list <- lapply(model_name_vec,function(x){
    mod_retriever(x)
  })
  
  out_inter <- foreach::foreach(i = 1:n_mods,.combine = 'rbind')%do%{
    
    mod_file <- model_file_list[[i]]
    mod_name <- model_name_vec[i]
    
   foreach::foreach(j = 1:n_methods, .combine = 'rbind')%do%{
    

  
  
  stan_fit <- run_joint(stan_list = stan_list,
                        model_file = mod_file,
                        mod_name = mod_name,
                        init = init,
                        iter = iter,
                        fit_method = fit_method_vec[j],
                        n_chains = n_chains,
                        n_par_chains = n_par_chains,
                        iter_warmup = iter_warmup,
                        iter_sampling = iter_sampling,
                        index = index,
                        use_rstan = use_rstan)
  
  x_u_1_fit <- stan_fit$x_u_1_fit
  x_u_2_fit <- stan_fit$x_u_2_fit
  
  
  impute_list[[i]][[j]] <- stan_fit$x_u_imputed
  y_models[[i]][[j]] <- stan_fit$y_model_df
  
  rmse_df <- stan_fit$rmse_df
 
  rmse_df
  
  }
  
  }
  
 out <- list(rmse_df = out_inter,
             impute_list = impute_list,
             y_models = y_models)
 
 out
  
}


# simulate and run -------------------------

sim_and_run_once <- function(n_z = n_z,
                             n_x = n_x,
                             n_r = n_r,
                             n = n,
                             mu_z = mu_z,
                             sd_z = sd_z,
                             sd_x = sd_x,
                             sd_y = sd_y,
                             alpha_vec_x = alpha_vec_x,
                             is_missing = is_missing,
                             r_prob_vec = r_prob_vec,
                             rand_cmat = TRUE,
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
                             model_name_vec = model_name_vec,
                             init = 0.2,
                             iter = 20000,
                             fit_method_vec,
                             n_chains = 4,
                             n_par_chains = 4,
                             iter_warmup = 1000,
                             iter_sampling = 1000,
                             index = 1,
                             use_rstan = FALSE){
  
  sim_once <- stan_sim_once(n_z = n_z,
                            n_x = n_x,
                            n_r = n_r,
                            n = n,
                            mu_z = mu_z,
                            sd_z = sd_z,
                            sd_x = sd_x,
                            sd_y = sd_y,
                            alpha_vec_x = alpha_vec_x,
                            is_missing = is_missing,
                            r_prob_vec = r_prob_vec,
                            rand_cmat = rand_cmat,
                            prob_cmat = prob_cmat,
                            rand_beta = rand_beta,
                            beta_sd = beta_sd,
                            C_mat_x_z = C_mat_x_z,
                            C_mat_x_x = C_mat_x_x,
                            C_mat_r_x = C_mat_r_x,
                            C_mat_r_r = C_mat_r_r,
                            C_mat_y_x = C_mat_y_x,
                            Beta_mat_x_z = Beta_mat_x_z,
                            Beta_mat_x_x = Beta_mat_x_x,
                            Beta_mat_r_x = Beta_mat_r_x,
                            Beta_mat_r_r = Beta_mat_r_r,
                            Beta_mat_y_x = Beta_mat_y_x)
  
  stan_list <- sim_once$stan_list
  emp_sd_vec_x <- sim_once$emp_sd_vec_x
  
  
  
  
  try_df <- run_joint_many(stan_list = stan_list,
                           model_name_vec = model_name_vec,
                           init = init,
                           iter = iter,
                           fit_method_vec = fit_method_vec,
                           n_chains = n_chains,
                           n_par_chains = n_par_chains,
                           iter_warmup = iter_warmup,
                           iter_sampling = iter_sampling,
                           index = index,
                           use_rstan = use_rstan)
  
  
 out <- list(rmse_df = try_df$rmse_df,
             stan_list = stan_list,
             emp_sd_vec_x = emp_sd_vec_x,
             impute_list = try_df$impute_list)
 
 out
  
  
}

# sim and run many -------------------------

sim_and_run_many <- function(N,
                             n_z,
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
                             rand_cmat,
                             prob_cmat,
                             rand_beta,
                             beta_sd,
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
                             model_name_vec,
                             init = 0.2,
                             iter = 20000,
                             fit_method_vec,
                             n_chains = 4,
                             n_par_chains = 4,
                             iter_warmup = 1000,
                             iter_sampling = 1000,
                             use_rstan = FALSE){
  
stan_lists <- vector('list', length = N)
impute_lists <- vector('list',length = N)
  
 out_int <-  foreach::foreach(i = 1:N,
                              .combine = 'rbind',
                              .errorhandling = 'remove')%do%{
   
   
 inter <-   sim_and_run_once(n_z = n_z,
                    n_x = n_x,
                    n_r = n_r,
                    n = n,
                    mu_z = mu_z,
                    sd_z = sd_z,
                    sd_x = sd_x,
                    sd_y = sd_y,
                    alpha_vec_x = alpha_vec_x,
                    is_missing = is_missing,
                    r_prob_vec = r_prob_vec,
                    rand_cmat = rand_cmat,
                    prob_cmat = prob_cmat,
                    rand_beta = rand_beta,
                    beta_sd = beta_sd,
                    C_mat_x_z = C_mat_x_z,
                    C_mat_x_x = C_mat_x_x,
                    C_mat_r_x = C_mat_r_x,
                    C_mat_r_r = C_mat_r_r,
                    C_mat_y_x = C_mat_y_x,
                    Beta_mat_x_z = Beta_mat_x_z,
                    Beta_mat_x_x = Beta_mat_x_x,
                    Beta_mat_r_x = Beta_mat_r_x,
                    Beta_mat_r_r = Beta_mat_r_r,
                    Beta_mat_y_x = Beta_mat_y_x,
                    model_name_vec = model_name_vec,
                    init = init,
                    iter = iter,
                    fit_method_vec = fit_method_vec,
                    n_chains = n_chains,
                    n_par_chains = n_par_chains,
                    iter_warmup = iter_warmup,
                    iter_sampling = iter_sampling,
                    index = i,
                    use_rstan = use_rstan)
 
 stan_lists[[i]] <- inter$stan_list
 impute_lists[[i]] <- inter$impute_list
 rmse_df <- inter$rmse_df
 
 rmse_df
   
 }
  
 out <- list(stan_lists = stan_lists,
             impute_lists = impute_lists,
             rmse_df = out_int) 
 
 out
}