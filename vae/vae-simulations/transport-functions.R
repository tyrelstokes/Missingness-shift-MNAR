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

mice_rmse_rf <- rbind(mice_rmse_source_rf,
                      mice_rmse_target_rf)

mice_rmse_mean <- rbind(mice_rmse_source_mean,
                        mice_rmse_target_mean)

mice_rmse_norm <- rbind(mice_rmse_source_norm,
                        mice_rmse_target_norm)

mice_rmse_ri <- rbind(mice_rmse_source_ri,
                      mice_rmse_target_ri)

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
