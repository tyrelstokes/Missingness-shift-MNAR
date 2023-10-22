

# Source Files----------------------
`%do%` <- foreach::`%do%`

source(here::here("vae/vae-simulations/sim-full-data-funs.R"))
source(here::here("vae/impute_models/stan-model-run-functions.R"))
source(here::here("vae/outcome-models/run-outcome-models.R"))
source(here::here("vae/vae-simulations/stan-simulation-prep.R"))
source(here::here("vae/impute_models/run_python_functions.R"))
source(here::here("vae/impute_models/run_mice.R"))

# realize function --------------------



# transport the models ----------------

transport_once <- function(y_df,
                           x_u_imputed,
                           x_o,
                           uncertainty = FALSE,
                           n_draws = 5000,
                           tol = .0001,
                           y,
                           full_data_lin_coefs_est,
                           linear_coefs){
 
  alpha <- get_quant(vb_summer = as.data.frame(y_df),
                     nm = "alpha")
  beta_o <- get_quant(vb_summer = as.data.frame(y_df),
                      nm = "beta_o")
  beta_u <- get_quant(vb_summer = as.data.frame(y_df),
                      nm = "beta_u")
  sigma <- get_quant(vb_summer = as.data.frame(y_df),
                     nm = "sigma")
  
  
  bias_est <- data.frame(bias_alpha = (alpha$mean - full_data_lin_coefs_est[1]),
                         bias_b1 = beta_o$mean - full_data_lin_coefs_est[2],
                         bias_b2 = beta_u$mean[1] - full_data_lin_coefs_est[3],
                         bias_b3 = beta_u$mean[2] - full_data_lin_coefs_est[4],
                         bias_sig = sigma$mean - full_data_lin_coefs_est[5])
  
  bias_est <- data.frame(vars = c("alpha_y","b1","b2","b3","sigma_y"),
                         est = c(alpha$mean,beta_o$mean,beta_u$mean,sigma$mean),
                         lm_est_full = full_data_lin_coefs_est,
                         lin_coefs = t(linear_coefs))
  
  bias_est$bias_est <- bias_est$lm_est_full-bias_est$est
  bias_est$bias_lin <- bias_est$lin_coefs - bias_est$est
  
  
  
  if(uncertainty == FALSE){
    y_pred <- alpha$mean + x_o*beta_o$mean + x_u_imputed%*%c(beta_u$mean)
    rmse <- sqrt(mean((y-y_pred)^2))
    out <- list(y_pred = y_pred,
                rmse = rmse,
                bias_est = bias_est)
    
    
  }else{
   
    
   y_pred_list1 <- lapply(1:n_draws,function(i){
     alpha_rl <- rnorm(1,alpha$mean, sd = alpha$sd)
     beta_o_rl <- rnorm(1,beta_o$mean, sd = beta_o$sd)
     beta_u_rl <- rnorm(2, beta_u$mean, sd = beta_u$sd)
     sigma_rl <- max(rnorm(1,sigma$mean,sd = sigma$sd),tol)
     alpha_rl + x_o*beta_o_rl + x_u_imputed%*%c(beta_u_rl)
   })
   
   y_pred_list2 <- lapply(1:n_draws,function(i){
     alpha_rl <- rnorm(1,alpha$mean, sd = alpha$sd)
     beta_o_rl <- rnorm(1,beta_o$mean, sd = beta_o$sd)
     beta_u_rl <- rnorm(2, beta_u$mean, sd = beta_u$sd)
     sigma_rl <- max(rnorm(1,sigma$mean,sd = sigma$sd),tol)
     rnorm(alpha_rl + x_o*beta_o_rl + x_u_imputed%*%c(beta_u_rl),sd = sigma_rl)
   })
  
  y_pred_mat <- matrix(unlist(y_pred_list1),ncol = n_draws)
  y_pred_mat2 <- matrix(unlist(y_pred_list2),ncol = n_draws)
  y_pred <- apply(y_pred_mat,1,mean)
  y_pred2 <- apply(y_pred_mat2,1,mean)
  
  rmse <- sqrt(mean((y-y_pred)^2))
  rmse2 <- sqrt(mean((y-y_pred2)^2))
  
  out <- list(y_pred = y_pred,
              y_pred2 = y_pred2,
              y_pred_mat = y_pred_mat,
              y_pred_mat2 = y_pred_mat2,
              rmse = rmse,
              rmse2 = rmse2,
              bias_est = bias_est)
  
  
  }
  
  
  out
    
}


transport_y <- function(y_models_source,
                        target_impute_lists,
                        uncertainty = FALSE,
                        n_draws = 5000,
                        tol = 0.0001,
                        x_o,
                        y,
                        model_name_vec,
                        fit_method_vec,
                        index = 1,
                        full_data_lin_coefs_est,
                        linear_coefs){
  
  n_mods <- length(y_models_source)
  n_fit_types <- length(y_models_source[[1]])
  
  
  
  
  y_pred_list <- vector("list",length = n_mods)
  bias_est_list <- vector("list", length = n_mods)
  bias_est_list_small <- vector("list",length = n_mods)

  
 rmse_y_df <-  foreach::foreach(i = 1:n_mods,.combine = 'rbind')%do%{
    
 inner_df <-    foreach::foreach(j = 1:n_fit_types,.combine = 'rbind')%do%{
      
      y_df <- y_models_source[[i]][[j]]
      x_u_imputed <- target_impute_lists[[i]][[j]]
      
    transport_y_list <-  transport_once(y_df = y_df,
                                        x_u_imputed = x_u_imputed,
                                        x_o = x_o,
                                        uncertainty = uncertainty,
                                        n_draws = n_draws,
                                        tol = tol,
                                        y = y,
                                        full_data_lin_coefs_est = full_data_lin_coefs_est,
                                        linear_coefs = linear_coefs
                                        ) 
    
    y_pred_list[[i]][[j]] <- transport_y_list$y_pred
    bias_est <- transport_y_list$bias_est
    
    bias_est$model_name <- model_name_vec[i]
    bias_est$fit_type <- fit_method_vec[j]
    bias_est$index <- index
    
    
    bias_est_list[[i]][[j]] <- bias_est
    
    rmse_out <- data.frame(rmse = transport_y_list$rmse)
    rmse_out$model <- model_name_vec[i]
    rmse_out$fit_type <- fit_method_vec[j]
    rmse_out$index <- index
    
    rmse_out
      
    }
   
   bias_est_list_small[[i]] <- do.call(rbind,bias_est_list[[i]]) 
   
   inner_df
    
 }
 
 bias_est_df <- do.call(rbind,bias_est_list_small)

  
 out <- list(y_pred_list = y_pred_list,
             rmse_y_df = rmse_y_df,
             bias_est_list = bias_est_list,
             bias_est_df = bias_est_df)
 
 out

  
}

# reformat python rmse

re_pyth_rmse <- function(rmse){
 rmse2 <-  rmse %>% dplyr::filter(type != "var 1")
  
  rmse2$type <- ifelse(rmse2$type == "var total",
                       "combined",
                       rmse2$type)
  
  rmse2
}

# combine rmse ----------------------------



combine_rmse <- function(pvae_rmse,
                         notmiwae_rmse,
                         rmse_df_impute_source,
                         rmse_df_impute_target,
                         mice_rmse,
                         gina_rmse,
                         mice_rmse_rf,
                         mice_rmse_mean,
                         mice_rmse_norm,
                         mice_rmse_ri){
  
  rmse_df_impute_source$data_type <- "source"
  rmse_df_impute_target$data_type <- "target"
  
  pvae2 <- re_pyth_rmse(rmse = pvae_rmse)
  
  not2 <- re_pyth_rmse(rmse = notmiwae_rmse)
  
  mice2 <- re_pyth_rmse(rmse = mice_rmse)
  
  gina2 <- re_pyth_rmse(rmse = gina_rmse)
  
  rf2 <- re_pyth_rmse(rmse = mice_rmse_rf)
  
  mean2 <- re_pyth_rmse(rmse = mice_rmse_mean)
  
  norm2 <- re_pyth_rmse(rmse = mice_rmse_norm)
  
  ri2 <- re_pyth_rmse(rmse = mice_rmse_ri)
  
  n_com_1  <- which(names(pvae2) %in% names(rmse_df_impute_source))
  n_com_2  <- which(names(rmse_df_impute_source) %in% names(pvae2))
  
  d1 <- pvae2[,n_com_1]
  d2 <- rmse_df_impute_source[,n_com_2]
  d3 <- rmse_df_impute_target[,n_com_2]
  d4 <- not2[,n_com_1]
  d5 <- mice2[,n_com_1]
  d6 <- gina2[,n_com_1]
  d7 <- rf2[,n_com_1]
  d8 <- mean2[,n_com_1]
  d9 <- norm2[,n_com_1]
  d10 <- ri2[,n_com_1]
  
  rmse_comb <- rbind(d1,
                     d2,
                     d3,
                     d4,
                     d5,
                     d6,
                     d7,
                     d8,
                     d9,
                     d10)
  
  
  rmse_comb
}

# simulate and run -------------------------

sim_and_run_da_once <- function(n_z,
                             n_x,
                             n_r,
                             n,
                             target_prop = .2,
                             mu_z,
                             sd_z,
                             sd_x,
                             sd_y,
                             alpha_vec_x,
                             is_missing,
                             r_prob_vec,
                             rand_cmat = TRUE,
                             prob_cmat = 0.7,
                             rand_beta = TRUE,
                             beta_sd = 1,
                             C_mat_x_z_source = NULL,
                             C_mat_x_x_source = NULL,
                             C_mat_r_x_source = NULL,
                             C_mat_r_r_source = NULL,
                             C_mat_y_x_source = NULL,
                             Beta_mat_x_z_source = NULL,
                             Beta_mat_x_x_source = NULL,
                             Beta_mat_r_x_source = NULL,
                             Beta_mat_r_r_source = NULL,
                             Beta_mat_y_x_source = NULL,
                             C_mat_x_z_target = NULL,
                             C_mat_x_x_target = NULL,
                             C_mat_r_x_target = NULL,
                             C_mat_r_r_target = NULL,
                             C_mat_y_x_target = NULL,
                             Beta_mat_x_z_target = NULL,
                             Beta_mat_x_x_target = NULL,
                             Beta_mat_r_x_target = NULL,
                             Beta_mat_r_r_target = NULL,
                             Beta_mat_y_x_target = NULL,
                             model_name_vec,
                             init = 0.2,
                             iter = 20000,
                             fit_method_vec,
                             n_chains = 4,
                             n_par_chains = 4,
                             iter_warmup = 1000,
                             iter_sampling = 1000,
                             index = 1,
                             use_rstan = FALSE,
                             n_draws = 5000,
                             tol = 0.0001,
                             uncertainty = FALSE,
                             use_python = FALSE,
                             pvae = FALSE,
                             x_link_fun = "identity",
                             y_link_fun = "identity",
                             n_nodes_x = c(2,2),
                             n_layers_x = 2,
                             n_nodes_y = c(2,2),
                             n_layers_y = 2,
                             act_fun_y = "relu",
                             final_act_y = "linear",
                             act_fun_x = "relu",
                             final_act_x = "linear"){
  
  
  
  if(rand_beta == TRUE){
    
    Beta_mat_x_z <- matrix(rnorm(n_z*n_x, sd = beta_sd),nrow = n_z)
    Beta_mat_x_x <- matrix(rnorm(n_x*n_x,sd = beta_sd),nrow = n_x)
    Beta_mat_r_x <- matrix(rnorm(n_r*n_x,sd = beta_sd),nrow = n_r)
    Beta_mat_r_r <- matrix(rnorm(n_r*n_r,sd = beta_sd),nrow = n_r)
    Beta_mat_y_x <- matrix(rnorm(1*n_r,sd = beta_sd),nrow = n_x) 
    
    Beta_mat_x_z_source <- Beta_mat_x_z
    Beta_mat_x_x_source <- Beta_mat_x_x
    Beta_mat_r_x_source <- Beta_mat_r_x
    Beta_mat_r_r_source <- Beta_mat_r_r
    Beta_mat_y_x_source <- Beta_mat_y_x
    
    Beta_mat_x_z_target <- Beta_mat_x_z
    Beta_mat_x_x_target <- Beta_mat_x_x
    Beta_mat_r_x_target <- Beta_mat_r_x
    Beta_mat_r_r_target <- Beta_mat_r_r
    Beta_mat_y_x_target <- Beta_mat_y_x
    
    
    
  }
  
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
    
    
    C_mat_x_z_source <- C_mat_x_z 
    C_mat_x_x_source <- C_mat_x_x 
    C_mat_r_x_source <- C_mat_r_x 
    C_mat_r_r_source <- C_mat_r_r 
    C_mat_y_x_source <- C_mat_y_x 
    
    C_mat_x_z_target <- C_mat_x_z 
    C_mat_x_x_target <- C_mat_x_x 
    C_mat_r_x_target <- C_mat_r_x 
    C_mat_r_r_target <- C_mat_r_r 
    C_mat_y_x_target <- C_mat_y_x  
    
  }
  
  
  
  
  
  sim_source <- stan_sim_once(n_z = n_z,
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
                            rand_cmat = FALSE,
                            prob_cmat = prob_cmat,
                            rand_beta = FALSE,
                            beta_sd = beta_sd,
                            C_mat_x_z = C_mat_x_z_source,
                            C_mat_x_x = C_mat_x_x_source,
                            C_mat_r_x = C_mat_r_x_source,
                            C_mat_r_r = C_mat_r_r_source,
                            C_mat_y_x = C_mat_y_x_source,
                            Beta_mat_x_z = Beta_mat_x_z_source,
                            Beta_mat_x_x = Beta_mat_x_x_source,
                            Beta_mat_r_x = Beta_mat_r_x_source,
                            Beta_mat_r_r = Beta_mat_r_r_source,
                            Beta_mat_y_x = Beta_mat_y_x_source,
                            x_link_fun = x_link_fun,
                            y_link_fun = y_link_fun,
                            n_nodes_x = n_nodes_x,
                            n_layers_x = n_layers_x,
                            n_nodes_y = n_nodes_y,
                            n_layers_y = n_layers_y,
                            act_fun_y = act_fun_y,
                            final_act_y = final_act_y,
                            act_fun_x = act_fun_x,
                            final_act_x = final_act_x)
  
  print(paste("simulated source # ",index))
  
  stan_list_source <- sim_source$stan_list
  emp_sd_vec_x_source <- sim_source$emp_sd_vec_x
  
  observed_data_source <- sim_source$observed_data
  full_data_source <- sim_source$full_data
  
 full_data_lin_coefs_est_source <- sim_source$full_data_lin_coefs_est
 linear_coefs_source <- sim_source$linear_coefs

  
  try_df <- run_joint_many(stan_list = stan_list_source,
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
  
  print(paste("fitted joint models # ",index))
  
  
  rmse_df_impute_source <- try_df$rmse_df
  impute_list_source <- try_df$impute_list
  y_models_source <- try_df$y_models
  
  if(pvae == TRUE){
  pvae_df_source <- run_pvae(observed_data = observed_data_source,
                             full_data = full_data_source,
                             data_type = "source",
                             index = index)

  pvae_rmse_source <- pvae_df_source$rmse_df
  pvae_imputed_source <- pvae_df_source$imputed_data
  pvae_y_model_source <- pvae_df_source$y_model_df
  
  print(paste("fit pvae #",index))
  
  n_impute <- length(impute_list_source)
  impute_list_source[[n_impute + 1]] <- list(pvae_imputed_source)
  y_models_source[[n_impute + 1]] <- list(pvae_y_model_source)
  
  
  notmiwae_df_source <- run_notmiwae(observed_data = observed_data_source,
                           full_data = full_data_source,
                           data_type = "source",
                           index = index)
  
  print(paste("fit notmiwae # ",index))
  
  notmiwae_rmse_source <- notmiwae_df_source$rmse_df
  notmiwae_imputed_source <- notmiwae_df_source$imputed_data
  notmiwae_y_model_source <- notmiwae_df_source$y_model_df
  
  impute_list_source[[n_impute + 2]] <- list(notmiwae_imputed_source)
  y_models_source[[n_impute + 2]] <- list(notmiwae_y_model_source)
  
  mice_df_source <- mice_function(observed_data = observed_data_source,
                             full_data = full_data_source,
                             data_type = "source",
                             index = index,
                             method = "pmm")
  
  
  print(paste("fit mice #",index))
  
  mice_rmse_source <- mice_df_source$rmse_df
  mice_imputed_source <- mice_df_source$imputed_data
  mice_y_model_source <- mice_df_source$y_model_df
  
  
  impute_list_source[[n_impute + 3]] <- list(mice_imputed_source)
  y_models_source[[n_impute + 3]] <- list(mice_y_model_source)
  
  gina_df_source <- run_gina(observed_data = observed_data_source,
                              full_data = full_data_source,
                              data_type = "source",
                              index = index)
  
  gina_rmse_source <- gina_df_source$rmse_df
  gina_imputed_source <- gina_df_source$imputed_data
  gina_y_model_source <- gina_df_source$y_model_df
  
  impute_list_source[[n_impute + 4]] <- list(gina_imputed_source)
  y_models_source[[n_impute + 4]] <- list(gina_y_model_source)
  
  print(paste("fit gina target #",index))
  
  mice_df_source_rf <- mice_function(observed_data = observed_data_source,
                                  full_data = full_data_source,
                                  data_type = "source",
                                  index = index,
                                  method = "rf")
  
  print(paste("fit mice rf #",index))
  
  mice_rmse_source_rf <- mice_df_source_rf$rmse_df
  mice_imputed_source_rf <- mice_df_source_rf$imputed_data
  mice_y_model_source_rf <- mice_df_source_rf$y_model_df
  
  
  impute_list_source[[n_impute + 5]] <- list(mice_imputed_source_rf)
  y_models_source[[n_impute + 5]] <- list(mice_y_model_source_rf)
  
  mice_df_source_mean <- mice_function(observed_data = observed_data_source,
                                     full_data = full_data_source,
                                     data_type = "source",
                                     index = index,
                                     method = "mean")
  
  print(paste("fit mice mean #",index))
  
  mice_rmse_source_mean  <- mice_df_source_mean$rmse_df
  mice_imputed_source_mean  <- mice_df_source_mean$imputed_data
  mice_y_model_source_mean  <- mice_df_source_mean$y_model_df
  
  
  impute_list_source[[n_impute + 6]] <- list(mice_imputed_source_mean)
  y_models_source[[n_impute + 6]] <- list(mice_y_model_source_mean)
  
  mice_df_source_norm <- mice_function(observed_data = observed_data_source,
                                       full_data = full_data_source,
                                       data_type = "source",
                                       index = index,
                                       method = "norm")
  
  print(paste("fit mice norm #",index))
  
  mice_rmse_source_norm  <- mice_df_source_norm$rmse_df
  mice_imputed_source_norm  <- mice_df_source_norm$imputed_data
  mice_y_model_source_norm  <- mice_df_source_norm$y_model_df
  
  
  impute_list_source[[n_impute + 7]] <- list(mice_imputed_source_norm)
  y_models_source[[n_impute + 7]] <- list(mice_y_model_source_norm)
  
  mice_df_source_ri <- mice_function(observed_data = observed_data_source,
                                       full_data = full_data_source,
                                       data_type = "source",
                                       index = index,
                                       method = "ri")
  
  print(paste("fit mice ri #",index))
  
  mice_rmse_source_ri  <- mice_df_source_ri$rmse_df
  mice_imputed_source_ri  <- mice_df_source_ri$imputed_data
  mice_y_model_source_ri  <- mice_df_source_ri$y_model_df
  
  
  impute_list_source[[n_impute + 8]] <- list(mice_imputed_source_ri)
  y_models_source[[n_impute + 8]] <- list(mice_y_model_source_ri)
  
  }
  
  n_target <- floor(n*target_prop)
  
  sim_target <- stan_sim_once(n = n_target,
                              n_z = n_z,
                              n_x = n_x,
                              n_r = n_r,
                              mu_z = mu_z,
                              sd_z = sd_z,
                              sd_x = sd_x,
                              sd_y = sd_y,
                              alpha_vec_x = alpha_vec_x,
                              is_missing = is_missing,
                              r_prob_vec = r_prob_vec,
                              rand_cmat = FALSE,
                              prob_cmat = prob_cmat,
                              rand_beta = FALSE,
                              beta_sd = beta_sd,
                              C_mat_x_z = C_mat_x_z_target,
                              C_mat_x_x = C_mat_x_x_target,
                              C_mat_r_x = C_mat_r_x_target,
                              C_mat_r_r = C_mat_r_r_target,
                              C_mat_y_x = C_mat_y_x_target,
                              Beta_mat_x_z = Beta_mat_x_z_target,
                              Beta_mat_x_x = Beta_mat_x_x_target,
                              Beta_mat_r_x = Beta_mat_r_x_target,
                              Beta_mat_r_r = Beta_mat_r_r_target,
                              Beta_mat_y_x = Beta_mat_y_x_target,
                              x_link_fun = x_link_fun,
                              y_link_fun = y_link_fun,
                              n_nodes_x = n_nodes_x,
                              n_layers_x = n_layers_x,
                              n_nodes_y = n_nodes_y,
                              n_layers_y = n_layers_y,
                              act_fun_y = act_fun_y,
                              final_act_y = final_act_y,
                              act_fun_x = act_fun_x,
                              final_act_x = final_act_x)
  
  print(paste("simulated target #",index))
  
  stan_list_target <- sim_target$stan_list
  emp_sd_vec_x_target <- sim_target$emp_sd_vec_x
  
  observed_data_target <- sim_target$observed_data
  full_data_target <- sim_target$full_data
  
  full_data_lin_coefs_est_target <- sim_target$full_data_lin_coefs_est
  linear_coefs_target <- sim_target$linear_coefs
  
  try_df_target <- run_joint_many(stan_list = stan_list_target,
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
  
  
  print(paste("fit stan models target #",index))
  
 rmse_df_impute_target <- try_df_target$rmse_df
  
  target_impute_lists <- try_df_target$impute_list
  
  x_o_transport <- sim_target$stan_list$x_o
  y_transport <- sim_target$stan_list$y
  y_models_target <- try_df_target$y_models
  
  
  if(pvae == TRUE){
  pvae_df_target <- run_pvae(observed_data = observed_data_target,
                             full_data = full_data_target,
                             data_type = "target",
                             index = index)
  
  
  pvae_rmse_target <- pvae_df_target$rmse_df
  pvae_imputed_target <- pvae_df_target$imputed_data
  pvae_y_model_target <- pvae_df_target$y_model_df
  print(paste("fit pvae target #",index))
  
  n_impute <- length(target_impute_lists)
  target_impute_lists[[n_impute + 1]] <- list(pvae_imputed_target)
  y_models_target[[n_impute + 1]] <- list(pvae_y_model_target)
  
  notmiwae_df_target <- run_notmiwae(observed_data = observed_data_target,
                                     full_data = full_data_target,
                                     data_type = "target",
                                     index = index)
  
  print(paste("fit notmiwae target #",index))
  
  notmiwae_rmse_target <- notmiwae_df_target$rmse_df
  notmiwae_imputed_target <- notmiwae_df_target$imputed_data
  notmiwae_y_model_target <- notmiwae_df_target$y_model_df
  
  target_impute_lists[[n_impute + 2]] <- list(notmiwae_imputed_target)
  y_models_target[[n_impute + 2]] <- list(notmiwae_y_model_target)
  
  mice_df_target <- mice_function(observed_data = observed_data_target,
                                  full_data = full_data_target,
                                  data_type = "target",
                                  index = index)
  
  mice_rmse_target <- mice_df_target$rmse_df
  mice_imputed_target <- mice_df_target$imputed_data
  mice_y_model_target <- mice_df_target$y_model_df
  
  print(paste("fit mice target #",index))
  
  
  target_impute_lists[[n_impute + 3]] <- list(mice_imputed_target)
  y_models_target[[n_impute + 3]] <- list(mice_y_model_target)
  
  gina_df_target <- run_gina(observed_data = observed_data_target,
                             full_data = full_data_target,
                             data_type = "target",
                             index = index)
  
  gina_rmse_target <- gina_df_target$rmse_df
  gina_imputed_target <- gina_df_target$imputed_data
  gina_y_model_target <- gina_df_target$y_model_df
  
  target_impute_lists[[n_impute + 4]] <- list(gina_imputed_target)
  y_models_target[[n_impute + 4]] <- list(gina_y_model_target)
  
  
  print(paste("fit gina target #",index))
  
  
  mice_df_target_rf <- mice_function(observed_data = observed_data_target,
                                     full_data = full_data_target,
                                     data_type = "target",
                                     index = index,
                                     method = "rf")
  
  print(paste("fit mice rf #",index))
  
  mice_rmse_target_rf <- mice_df_target_rf$rmse_df
  mice_imputed_target_rf <- mice_df_target_rf$imputed_data
  mice_y_model_target_rf <- mice_df_target_rf$y_model_df
  
  target_impute_lists[[n_impute + 5]] <- list(mice_imputed_target_rf)
  y_models_target[[n_impute + 5]] <- list(mice_y_model_target_rf)
  
  mice_df_target_mean <- mice_function(observed_data = observed_data_target,
                                       full_data = full_data_target,
                                       data_type = "target",
                                       index = index,
                                       method = "mean")
  
  print(paste("fit mice mean target #",index))
  
  mice_rmse_target_mean  <- mice_df_target_mean$rmse_df
  mice_imputed_target_mean  <- mice_df_target_mean$imputed_data
  mice_y_model_target_mean  <- mice_df_target_mean$y_model_df
  
  
  target_impute_lists[[n_impute + 6]] <- list(mice_imputed_target_mean)
  y_models_target[[n_impute + 6]] <- list(mice_y_model_target_mean)
  
  mice_df_target_norm <- mice_function(observed_data = observed_data_target,
                                       full_data = full_data_target,
                                       data_type = "target",
                                       index = index,
                                       method = "norm")
  
  print(paste("fit mice norm target #",index))
  
  mice_rmse_target_norm  <- mice_df_target_norm$rmse_df
  mice_imputed_target_norm  <- mice_df_target_norm$imputed_data
  mice_y_model_target_norm  <- mice_df_target_norm$y_model_df
  
  
  target_impute_lists[[n_impute + 7]] <- list(mice_imputed_target_norm)
  y_models_target[[n_impute + 7]] <- list(mice_y_model_target_norm)
  
  mice_df_target_ri <- mice_function(observed_data = observed_data_target,
                                     full_data = full_data_target,
                                     data_type = "target",
                                     index = index,
                                     method = "ri")
  
  print(paste("fit mice ri #",index))
  
  mice_rmse_target_ri  <- mice_df_target_ri$rmse_df
  mice_imputed_target_ri  <- mice_df_target_ri$imputed_data
  mice_y_model_target_ri  <- mice_df_target_ri$y_model_df
  
  
  target_impute_lists[[n_impute + 8]] <- list(mice_imputed_target_ri)
  y_models_target[[n_impute + 8]] <- list(mice_y_model_target_ri)
  
  
  model_name_vec_2 <- c(model_name_vec,
                        "pvae",
                        "notmiwae",
                        "mice pmm",
                        "gina",
                        "mice rf",
                        "mean",
                        "mice lm",
                        "mice ri")
  
  y_predictions <- transport_y(y_models_source = y_models_source,
                               target_impute_lists = target_impute_lists,
                               uncertainty = uncertainty,
                               n_draws = n_draws,
                               tol = tol,
                               x_o = x_o_transport,
                               y = y_transport,
                               model_name_vec = model_name_vec_2,
                               fit_method_vec = fit_method_vec,
                               index = index,
                               full_data_lin_coefs_est = full_data_lin_coefs_est_target,
                               linear_coefs=  linear_coefs_target)
  
  print(paste("fit y prediction models #", index))
  
  pvae_rmse <- rbind(pvae_rmse_source,
                     pvae_rmse_target)
  
  notmiwae_rmse <- rbind(notmiwae_rmse_source,
                         notmiwae_rmse_target)
  
  mice_rmse <- rbind(mice_rmse_source,
                     mice_rmse_target)
  
  gina_rmse <- rbind(gina_rmse_source,
                     gina_rmse_target)
  
  mice_rmse_rf <- rbind(mice_rmse_source_rf,
                        mice_rmse_target_rf)
  
  mice_rmse_mean <- rbind(mice_rmse_source_mean,
                        mice_rmse_target_mean)
  
  mice_rmse_norm <- rbind(mice_rmse_source_norm,
                          mice_rmse_target_norm)
  
  mice_rmse_ri <- rbind(mice_rmse_source_ri,
                          mice_rmse_target_ri)
  
 rmse_x_comb <-  combine_rmse(pvae_rmse = pvae_rmse,
                              notmiwae_rmse = notmiwae_rmse,
                              rmse_df_impute_source = rmse_df_impute_source,
                              rmse_df_impute_target = rmse_df_impute_target,
                              mice_rmse = mice_rmse,
                              gina_rmse = gina_rmse,
                              mice_rmse_rf = mice_rmse_rf,
                              mice_rmse_mean = mice_rmse_mean,
                              mice_rmse_norm = mice_rmse_norm,
                              mice_rmse_ri = mice_rmse_ri)
 


  
  }else{
  
  y_predictions <- transport_y(y_models_source = y_models_source,
                            target_impute_lists = target_impute_lists,
                            uncertainty = uncertainty,
                            n_draws = n_draws,
                            tol = tol,
                            x_o = x_o_transport,
                            y = y_transport,
                            model_name_vec = model_name_vec,
                            fit_method_vec = fit_method_vec,
                            index = index,
                            full_data_lin_coefs_est = full_data_lin_coefs_est_target,
                            linear_coefs=  linear_coefs_target)
  

  
  }
  
  print(paste("finished y transport predictions #",index))
  
  y_pred_list <- y_predictions$y_pred_list
  rmse_df_y <- y_predictions$rmse_y_df
  bias_est_list <- y_predictions$bias_est_list

  bias_est_df <- y_predictions$bias_est_df
  #bias_lin_df <- y_predictions$bias_lin_df
  
  y_models_target <- try_df_target$y_models
  

if(pvae == FALSE){
  pvae_rmse <- NULL
  notmiwae_rmse <- NULL
  mice_rmse <- NULL
  gina_rmse <- NULL
  rmse_x_comb <- rbind(rmse_df_impute_source,
                       rmse_df_impute_target)
}
  
  
  out <- list(rmse_df_impute_source = rmse_df_impute_source,
              rmse_df_impute_target =  rmse_df_impute_target,
              stan_list_source = stan_list_source,
              stan_list_target = stan_list_target,
              emp_sd_vec_x_source = emp_sd_vec_x_source,
              emp_sd_vec_x_target = emp_sd_vec_x_target,
              y_pred_list = y_pred_list,
              target_impute_lists = target_impute_lists,
              impute_list_source = impute_list_source,
              rmse_df_y =  rmse_df_y,
              y_models_target = y_models_target,
              y_models_source = y_models_source,
              pvae_rmse = pvae_rmse,
              rmse_x_comb = rmse_x_comb,
              notmiwae_rmse = notmiwae_rmse,
              mice_rmse = mice_rmse,
              bias_est_df = bias_est_df,
              gina_rmse = gina_rmse)
  
  print(paste("finished one full run of da: Index is #",index))
  
  out
  
  
}

# Simulate and run many domain adapt ---------------------


sim_and_run_da_many <- function(N = N,
                                n_z,
                                n_x,
                                n_r,
                                n,
                                target_prop = .2,
                                mu_z,
                                sd_z,
                                sd_x,
                                sd_y,
                                alpha_vec_x,
                                is_missing,
                                r_prob_vec,
                                rand_each_run = FALSE,
                                rand_cmat = TRUE,
                                prob_cmat = 0.7,
                                rand_beta = TRUE,
                                beta_sd = 1,
                                C_mat_x_z_source = NULL,
                                C_mat_x_x_source = NULL,
                                C_mat_r_x_source = NULL,
                                C_mat_r_r_source = NULL,
                                C_mat_y_x_source = NULL,
                                Beta_mat_x_z_source = NULL,
                                Beta_mat_x_x_source = NULL,
                                Beta_mat_r_x_source = NULL,
                                Beta_mat_r_r_source = NULL,
                                Beta_mat_y_x_source = NULL,
                                C_mat_x_z_target = NULL,
                                C_mat_x_x_target = NULL,
                                C_mat_r_x_target = NULL,
                                C_mat_r_r_target = NULL,
                                C_mat_y_x_target = NULL,
                                Beta_mat_x_z_target = NULL,
                                Beta_mat_x_x_target = NULL,
                                Beta_mat_r_x_target = NULL,
                                Beta_mat_r_r_target = NULL,
                                Beta_mat_y_x_target = NULL,
                                model_name_vec,
                                init = 0.2,
                                iter = 20000,
                                fit_method_vec,
                                n_chains = 4,
                                n_par_chains = 4,
                                iter_warmup = 1000,
                                iter_sampling = 1000,
                                use_rstan = FALSE,
                                n_draws = 5000,
                                tol = 0.0001,
                                uncertainty = FALSE,
                                pvae = FALSE,
                                x_link_fun = "identity",
                                y_link_fun = "identity",
                                n_nodes_x = c(2,2),
                                n_layers_x = 2,
                                n_nodes_y = c(2,2),
                                n_layers_y = 2,
                                act_fun_y = "relu",
                                final_act_y = "linear",
                                act_fun_x = "relu",
                                final_act_x = "linear"){
  
  beg_time <- Sys.time()
  
  stan_lists_source <- vector('list', length = N)
  stan_lists_target <- vector('list', length = N)
  
  impute_lists_source <- vector('list', length = N)
  impute_lists_target <- vector('list', length = N)
  
  y_models_source <- vector('list', length = N)
  y_models_target <- vector('list', length = N)
  
  rmse_impute <- vector('list',length = N)
  
  rmse_comb <- vector('list', length = N)
  
  bias_est_lists <- vector('list',length = N)
  
  
  if(rand_beta == TRUE){
    
    Beta_mat_x_z <- matrix(rnorm(n_z*n_x, sd = beta_sd),nrow = n_z)
    Beta_mat_x_x <- matrix(rnorm(n_x*n_x,sd = beta_sd),nrow = n_x)
    Beta_mat_r_x <- matrix(rnorm(n_r*n_x,sd = beta_sd),nrow = n_r)
    Beta_mat_r_r <- matrix(rnorm(n_r*n_r,sd = beta_sd),nrow = n_r)
    Beta_mat_y_x <- matrix(rnorm(1*n_r,sd = beta_sd),nrow = n_x) 
    
    Beta_mat_x_z_source <- Beta_mat_x_z
    Beta_mat_x_x_source <- Beta_mat_x_x
    Beta_mat_r_x_source <- Beta_mat_r_x
    Beta_mat_r_r_source <- Beta_mat_r_r
    Beta_mat_y_x_source <- Beta_mat_y_x
    
    Beta_mat_x_z_target <- Beta_mat_x_z
    Beta_mat_x_x_target <- Beta_mat_x_x
    Beta_mat_r_x_target <- Beta_mat_r_x
    Beta_mat_r_r_target <- Beta_mat_r_r
    Beta_mat_y_x_target <- Beta_mat_y_x
    
    
    
  }
  
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
    
    
    C_mat_x_z_source <- C_mat_x_z 
    C_mat_x_x_source <- C_mat_x_x 
    C_mat_r_x_source <- C_mat_r_x 
    C_mat_r_r_source <- C_mat_r_r 
    C_mat_y_x_source <- C_mat_y_x 
    
    C_mat_x_z_target <- C_mat_x_z 
    C_mat_x_x_target <- C_mat_x_x 
    C_mat_r_x_target <- C_mat_r_x 
    C_mat_r_r_target <- C_mat_r_r 
    C_mat_y_x_target <- C_mat_y_x  
    
  }
  

  
  
  out_int <-  foreach::foreach(i = 1:N,
                               .combine = 'rbind',
                               .errorhandling = 'remove')%do%{
                                 
        if(rand_each_run == FALSE){                         
        inter <-  sim_and_run_da_once(n_z = n_z,
                                      n_x = n_x,
                                      n_r = n_r,
                                      n = n,
                                      target_prop = target_prop,
                                      mu_z = mu_z,
                                      sd_z = sd_z,
                                      sd_x = sd_x,
                                      sd_y = sd_y,
                                      alpha_vec_x = alpha_vec_x,
                                      is_missing = is_missing,
                                      r_prob_vec = r_prob_vec,
                                      rand_cmat = FALSE,
                                      prob_cmat = prob_cmat,
                                      rand_beta = FALSE,
                                      beta_sd = beta_sd,
                                      C_mat_x_z_source = C_mat_x_z_source,
                                      C_mat_x_x_source = C_mat_x_x_source,
                                      C_mat_r_x_source = C_mat_r_x_source,
                                      C_mat_r_r_source = C_mat_r_r_source,
                                      C_mat_y_x_source = C_mat_y_x_source,
                                      Beta_mat_x_z_source = Beta_mat_x_z_source,
                                      Beta_mat_x_x_source = Beta_mat_x_x_source,
                                      Beta_mat_r_x_source = Beta_mat_r_x_source,
                                      Beta_mat_r_r_source = Beta_mat_r_r_source,
                                      Beta_mat_y_x_source = Beta_mat_y_x_source,
                                      C_mat_x_z_target = C_mat_x_z_target,
                                      C_mat_x_x_target = C_mat_x_x_target,
                                      C_mat_r_x_target = C_mat_r_x_target,
                                      C_mat_r_r_target = C_mat_r_r_target,
                                      C_mat_y_x_target = C_mat_y_x_target,
                                      Beta_mat_x_z_target = Beta_mat_x_z_target,
                                      Beta_mat_x_x_target = Beta_mat_x_x_target,
                                      Beta_mat_r_x_target = Beta_mat_r_x_target,
                                      Beta_mat_r_r_target = Beta_mat_r_r_target,
                                      Beta_mat_y_x_target = Beta_mat_y_x_target,
                                      model_name_vec = model_name_vec,
                                      init = init,
                                      iter = iter,
                                      fit_method_vec = fit_method_vec,
                                      n_chains = n_chains,
                                      n_par_chains = n_par_chains,
                                      iter_warmup = iter_warmup,
                                      iter_sampling = iter_sampling,
                                      index = i,
                                      use_rstan = use_rstan,
                                      n_draws = n_draws,
                                      tol = tol,
                                      uncertainty = uncertainty,
                                      pvae = pvae,
                                      x_link_fun = x_link_fun,
                                      y_link_fun = y_link_fun,
                                      n_nodes_x = n_nodes_x,
                                      n_layers_x = n_layers_x,
                                      n_nodes_y = n_nodes_y,
                                      n_layers_y = n_layers_y,
                                      act_fun_y = act_fun_y,
                                      final_act_y = final_act_y,
                                      act_fun_x = act_fun_x,
                                      final_act_x = final_act_x) 
        
        }else{
          
          inter <-  sim_and_run_da_once(n_z = n_z,
                                        n_x = n_x,
                                        n_r = n_r,
                                        n = n,
                                        target_prop = target_prop,
                                        mu_z = mu_z,
                                        sd_z = sd_z,
                                        sd_x = sd_x,
                                        sd_y = sd_y,
                                        alpha_vec_x = alpha_vec_x,
                                        is_missing = is_missing,
                                        r_prob_vec = r_prob_vec,
                                        rand_cmat = rand_cmat,
                                        prob_cmat = prob_cmat,
                                        rand_beta = TRUE,
                                        beta_sd = beta_sd,
                                        C_mat_x_z_source = C_mat_x_z_source,
                                        C_mat_x_x_source = C_mat_x_x_source,
                                        C_mat_r_x_source = C_mat_r_x_source,
                                        C_mat_r_r_source = C_mat_r_r_source,
                                        C_mat_y_x_source = C_mat_y_x_source,
                                        Beta_mat_x_z_source = Beta_mat_x_z_source,
                                        Beta_mat_x_x_source = Beta_mat_x_x_source,
                                        Beta_mat_r_x_source = Beta_mat_r_x_source,
                                        Beta_mat_r_r_source = Beta_mat_r_r_source,
                                        Beta_mat_y_x_source = Beta_mat_y_x_source,
                                        C_mat_x_z_target = C_mat_x_z_target,
                                        C_mat_x_x_target = C_mat_x_x_target,
                                        C_mat_r_x_target = C_mat_r_x_target,
                                        C_mat_r_r_target = C_mat_r_r_target,
                                        C_mat_y_x_target = C_mat_y_x_target,
                                        Beta_mat_x_z_target = Beta_mat_x_z_target,
                                        Beta_mat_x_x_target = Beta_mat_x_x_target,
                                        Beta_mat_r_x_target = Beta_mat_r_x_target,
                                        Beta_mat_r_r_target = Beta_mat_r_r_target,
                                        Beta_mat_y_x_target = Beta_mat_y_x_target,
                                        model_name_vec = model_name_vec,
                                        init = init,
                                        iter = iter,
                                        fit_method_vec = fit_method_vec,
                                        n_chains = n_chains,
                                        n_par_chains = n_par_chains,
                                        iter_warmup = iter_warmup,
                                        iter_sampling = iter_sampling,
                                        index = i,
                                        use_rstan = use_rstan,
                                        n_draws = n_draws,
                                        tol = tol,
                                        uncertainty = uncertainty,
                                        pvae = pvae,
                                        x_link_fun = x_link_fun,
                                        y_link_fun = y_link_fun,
                                        n_nodes_x = n_nodes_x,
                                        n_layers_x = n_layers_x,
                                        n_nodes_y = n_nodes_y,
                                        n_layers_y = n_layers_y,
                                        act_fun_y = act_fun_y,
                                        final_act_y = final_act_y,
                                        act_fun_x = act_fun_x,
                                        final_act_x = final_act_x)  
          
        }
        
        

        
        stan_lists_source[[i]] <-   inter$stan_list_source
        stan_lists_target[[i]] <-   inter$stan_list_target
        
        impute_lists_source[[i]] <- inter$impute_list_source
        impute_lists_target[[i]] <- inter$target_impute_lists
        
        rmse_df_impute_source <- inter$rmse_df_impute_source
        rmse_df_impute_source$data_source <- "source"
        rmse_df_impute_source$n <- n
        
        rmse_df_impute_target <- inter$rmse_df_impute_target
        rmse_df_impute_target$data_source <- "target"
        rmse_df_impute_target$n <- floor(n*target_prop)
        
        
        y_models_source[[i]] <- inter$y_models_source
        y_models_target[[i]] <- inter$y_models_target
        bias_est_lists[[i]] <- inter$bias_est_df
        
        impute_rmse <- rbind(rmse_df_impute_source,
                             rmse_df_impute_target)
        
        rmse_impute[[i]] <- impute_rmse
        
        rmse_comb[[i]] <- inter$rmse_x_comb
   
        rmse_df_y <- inter$rmse_df_y
       
       #rmse_df_y
       
       
       cur_time <- Sys.time()
       
       elapsed_time <- cur_time - beg_time
       
       avg_time <- elapsed_time/i
       
       est_remain_time <- (N-i)*avg_time
       
       print(paste("The total elapsed time is:",
                   paste(round(elapsed_time,3),
                         "minutes")))
       print(paste("A crude estimate of the remaining time is:",
                   paste(round(est_remain_time,3),
                         "minutes")))
       
       
       rmse_df_y
                                 
                               }
  
  if(N > 1){
  print(paste("The average time per loop was",
              paste(round(avg_time,2),
                    "minutes")))
  
  print(paste("The total time to complete",paste(N,paste("loops was:",
              paste(round(elapsed_time,2),
                    "minutes")))))
  
  }
  
  impute_rmse_df <- do.call(rbind,rmse_impute)
  bias_est_df_full <- do.call(rbind,bias_est_lists)

  
  rmse_x_df <- do.call(rbind,rmse_comb)
  
  out <- list(stan_lists_source = stan_lists_source,
              stan_lists_target = stan_lists_target,
              impute_lists_source = impute_lists_source,
              impute_lists_target = impute_lists_target,
              rmse_df_y = out_int,
              impute_rmse_df = impute_rmse_df,
              y_models_source = y_models_source,
              y_models_target = y_models_target,
              rmse_comb = rmse_comb,
              rmse_x_df = rmse_x_df,
              bias_est_df = bias_est_df_full) 
  
  out
  
  
  
}







