# python calc rmse ------------------


rmse_one_frame <- function(r,
                           x,
                           act,
                           i,
                           model_name,
                           data_type){
  
  indicate <- which(r ==0)
  
  x_ind <- x[indicate]
  act_ind <- act[indicate]
  
  rmse <- sqrt(mean((x_ind - act_ind)^2))
  pct_missing <- mean(r)
  var_ind <- i
  
  out_df <- data.frame(index = index,
                       type = paste0("var ",var_ind),
                       model_name = model_name,
                       fit_method = "vb",
                       data_type = data_type,
                       rmse = rmse,
                       pct_missing = pct_missing)
  out_df
  
}

rmse_calc <- function(observed_data,
                      full_data,
                      imputed_data,
                      model_name,
                      data_type,
                      index = 1){
  
  
  
  
  imputed_x1 <- imputed_data[,1]
  imputed_x2 <- imputed_data[,2]
  imputed_x3 <- imputed_data[,3]
  
  rmat <- data.frame(r_1 = observed_data$r_1,
                     r_2 = observed_data$r_2,
                     r_3 = observed_data$r_3)
  
  imputed_data <- data.frame(x1 = imputed_x1,
                             x2 = imputed_x2,
                             x3 = imputed_x3)
  
  actual_df <- data.frame(x1 = full_data$x_1,
                          x2 = full_data$x_2,
                          x3 = full_data$x_3)
  
  comb_list <- vector("list",length = 3)
  
  rmse_x <- foreach::foreach(i = 1:3, .combine = 'rbind')%do%{
    r <- rmat[,i]
    x <- imputed_data[,i]
    act <- actual_df[,i]
    
    
    comb_list[[i]] <- data.frame(x = x,
                                 act = act,
                                 r = r)
    
    
    out_df <- rmse_one_frame(r = r,
                             x = x,
                             act = act,
                             i = i,
                             model_name = model_name,
                             data_type = data_type)
    
  }
  
  comb_df <- do.call('rbind',comb_list)
  
  rmse_t <- rmse_one_frame(r = comb_df$r,
                           x = comb_df$x,
                           act = comb_df$act,
                           i = "total",
                           model_name = model_name,
                           data_type = data_type)
  
  rmse_df <- rbind(rmse_x,
                   rmse_t)
  
  
  out <- list(rmse_df = rmse_df,
              imputed_data = imputed_data[,2:3])
  
  out
}


#################################################
# Outcome model part

fit_outcome_model <- function(fit_outcome = FALSE,
                              y,
                              x_u,
                              x_o,
                              R,
                              fit_method = "vb",
                              init = 0.2,
                              iter = 2000,
                              n_chains = 4,
                              n_par_chains = 4,
                              iter_warmup = 1000,
                              iter_sampling = 1000,
                              rmse_list){
  
  if(fit_outcome ==TRUE){
    
    out_mod <- run_outcome_model(y = y,
                                 x_u = x_u,
                                 x_o = x_o,
                                 R = R,
                                 fit_method = fit_method,
                                 init = init,
                                 iter = iter,
                                 n_chains = n_chains,
                                 n_par_chains = n_par_chains,
                                 iter_warmup = iter_warmup,
                                 iter_sampling = iter_sampling
    )
    
    param_df <-  out_mod$param_df
    
    y_model_df <- param_df %>% dplyr::filter(complete_case ==FALSE)
    
    y_preds_full <- out_mod$y_pred
    y_preds <- y_preds_full$mean
    
    rmse_y <- sqrt(mean((y - y_preds)^2))
    
    rmse_df <- rmse_list$rmse_df
    
    
    
    out <- list(y_model_df = y_model_df,
                y_preds = y_preds,
                y_preds_full = y_preds_full,
                rmse_y = rmse_y,
                imputed_data = x_u,
                rmse_df = rmse_df)
    
    
  }else{
    
    rmse_df <- rmse_list$rmse_df
    
    
    
    out <- list(imputed_data = x_u_imputed,
                rmse_df = rmse_df)
    
    
    
  }
  
  out
  
}

###################################################
##################################################

#python_int <- "vae/env/bin/python"
#python_int <- "python"

# Run Hyungrok's python script -----------------

run_pvae <- function(observed_data,
                     full_data,
                     data_type,
                     index = 1,
                     python_interpretter = python_int,
                     fit_outcome = FALSE){
  
  write.csv(observed_data,"vae/python-data/pvae_input.csv")
  
  #system("source vae/env/bin/activate")
  #py_command <- "vae/python/main.py --input_file vae/python-data/pvae_input.csv --save_dir vae/python-data"
  py_command <- "vae/imputation/main.py --input_file vae/python-data/pvae_input.csv --method pvae --save_dir vae/python-data"
  
  script_text <- paste(python_interpretter,
                       py_command)
  
  system(script_text)
  
  pvae_imputed <- read.csv("vae/python-data/pvae_input_pvae_imputed.csv")
  
  pvae_imputed_small <- pvae_imputed[,c("xobs_1","xobs_2","xobs_3")]
  
  rmse_list <- rmse_calc(observed_data = observed_data,
                         full_data = full_data,
                         imputed_data = pvae_imputed_small,
                         model_name = "pvae",
                         data_type = data_type,
                         index = index)
  
  x_u_imputed <- cbind(pvae_imputed$xobs_2,
                       pvae_imputed$xobs_3)
  
  
  
  out <-      fit_outcome_model(fit_outcome = TRUE,
                                y = observed_data$y,
                                x_u = x_u_imputed,
                                x_o = observed_data$xobs_1,
                                R = cbind(observed_data$r_2,
                                          observed_data$r_3),
                                fit_method = "vb",
                                init = 0.2,
                                iter = 2000,
                                n_chains = 4,
                                n_par_chains = 4,
                                iter_warmup = 1000,
                                iter_sampling = 1000,
                                rmse_list = rmse_list)
  
  
  
  
  out
  
}

# run notMIWAE ----------------------------

run_notmiwae <- function(observed_data,
                         full_data,
                         data_type,
                         index = 1,
                         python_interpretter = python_int){
  
  write.csv(observed_data,"vae/python-data/pvae_input.csv")
  write.csv(full_data, "vae/python-data/full_data.csv")
  
  #py_command <- "vae/notMIWAE/notmiwae.py --fdata vae/python-data/full_data.csv --mdata vae/python-data/pvae_input.csv"
  
  py_command <- "vae/imputation/main.py --input_file vae/python-data/pvae_input.csv --method notmiwae --save_dir vae/python-data"
  #system("source vae/env/bin/activate")
  script_text <- paste(python_interpretter,
                       py_command)
  system(script_text)
  
  pvae_imputed <- read.csv("vae/python-data/pvae_input_notmiwae_imputed.csv")
  
  rmse_list <- rmse_calc(observed_data = observed_data,
                         full_data = full_data,
                         imputed_data = pvae_imputed,
                         model_name = "notMIWAE",
                         data_type = data_type,
                         index = index)
  
  x_u_imputed <- cbind(pvae_imputed[,2],
                       pvae_imputed[,3])
  
  out <-      fit_outcome_model(fit_outcome = TRUE,
                                y = observed_data$y,
                                x_u = x_u_imputed,
                                x_o = observed_data$xobs_1,
                                R = cbind(observed_data$r_2,
                                          observed_data$r_3),
                                fit_method = "vb",
                                init = 0.2,
                                iter = 2000,
                                n_chains = 4,
                                n_par_chains = 4,
                                iter_warmup = 1000,
                                iter_sampling = 1000,
                                rmse_list = rmse_list)
  
  
  
  
  out
  
  
  
}


# run mnar-pvae ----------------

run_gina <- function(observed_data,
                     full_data,
                     data_type,
                     index = 1,
                     python_interpretter = python_int){
  
  observed_data_small <- observed_data[,c("xobs_1","xobs_2","xobs_3")]
  
  write.csv(observed_data_small,"vae/python-data/gina_input.csv")
  
  #py_command <- "vae/gina/mnar_pvae_main.py --input_file vae/python-data/gina_input.csv --save_dir vae/python-data"
  py_command <- "vae/imputation/main.py --input_file vae/python-data/gina_input.csv --method gina --save_dir vae/python-data"
  #system("source vae/env/bin/activate")
  script_text <- paste(python_interpretter,
                       py_command)
  #system("source vae/env/bin/activate")
  
  
  system(script_text)
  
  pvae_imputed <- read.csv("vae/python-data/gina_input_gina_imputed.csv")
  
  pvae_imputed_small <- pvae_imputed[,c("xobs_1","xobs_2","xobs_3")]
  
  rmse_list <- rmse_calc(observed_data = observed_data,
                         full_data = full_data,
                         imputed_data = pvae_imputed_small,
                         model_name = "gina",
                         data_type = data_type,
                         index = index)
  
  x_u_imputed <- cbind(pvae_imputed$xobs_2,
                       pvae_imputed$xobs_3)
  
  out <-      fit_outcome_model(fit_outcome = TRUE,
                                y = observed_data$y,
                                x_u = x_u_imputed,
                                x_o = observed_data$xobs_1,
                                R = cbind(observed_data$r_2,
                                          observed_data$r_3),
                                fit_method = "vb",
                                init = 0.2,
                                iter = 2000,
                                n_chains = 4,
                                n_par_chains = 4,
                                iter_warmup = 1000,
                                iter_sampling = 1000,
                                rmse_list = rmse_list)
  
  
  
  
  
  out
  
}
