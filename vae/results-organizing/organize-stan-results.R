# Housekeeping
`%>%` <- dplyr::`%>%` 
source(here::here("vae/outcome-models/run-outcome-models.R"))
# Comp function ------------------

comp_fun <- function(truth,
                     fit,
                     exp,
                     var,
                     index){
  df <- data.frame(truth = truth,
                   fit = fit,
                   exp = exp,
                   index = index) %>%
    dplyr::mutate(diff = fit - truth,
                  variable = var,
                  bias = fit - exp)
}

# extract function --------------

extract_function <- function(fitted_model,
                             mod_name,
                             fit_type,
                             index,
                             use_rstan = FALSE,
                             outcome_pred = FALSE,
                             y = NULL,
                             x_o = NULL,
                             R = NULL){
  
  if(use_rstan == FALSE){
  vb_sum <- fitted_model$summary()
  }else{
    
    vb_sum <- as.data.frame(rstan::summary(fitted_model)[[1]])
    vb_sum$variable <- rownames(vb_sum)
  }
  
  x_u_1_fit <- get_quant(vb_summer = vb_sum, nm = "x_u_miss_1")
  x_u_2_fit <- get_quant(vb_summer = vb_sum, nm = "x_u_miss_2")
  
  x_u_1_true <- get_quant(vb_summer = vb_sum, nm = "x_u_true_1")
  x_u_2_true <- get_quant(vb_summer = vb_sum, nm = "x_u_true_2")
  
  exp_x_u_1 <- get_quant(vb_summer = vb_sum, nm = "exp_x_u_1")
  exp_x_u_2 <- get_quant(vb_summer = vb_sum, nm = "exp_x_u_2")
  
  x_u_re_1 <- get_quant(vb_summer = vb_sum, nm = "x_u_re_1")
  x_u_re_2 <- get_quant(vb_summer = vb_sum, nm = "x_u_re_2")
  

  
  if(mod_name %in% c("Z model", "Joint Model")){
    
    y_preds_full <- vb_sum[grepl("pred",vb_sum$variable),]
    y_preds <- y_preds_full$mean
    
    
    alpha_y <- get_quant(vb_summer = vb_sum, nm = "alpha_y")
    beta_u <- get_quant(vb_summer = vb_sum, nm = "beta_u")
    beta_o <- get_quant(vb_summer = vb_sum, nm = "beta_o")
    sigma_y <- get_quant(vb_summer = vb_sum, nm = "sigma_y")

    y_model_df <- rbind(alpha_y,
                        beta_u,
                        beta_o,
                        sigma_y)
    
  }else{
    if(outcome_pred ==FALSE){
    y_preds <- NULL
    y_preds_full <- NULL
    y_model_df <- NULL
    }else{
      
      x_u_re <- cbind(x_u_re_1$mean,
                      x_u_re_2$mean)
      
      
      out_mod <- run_outcome_model(y = y,
                        x_u = x_u_re,
                        x_o = x_o,
                        R = R,
                        fit_method = "vb",
                        init = 0.2,
                        iter = 20000,
                        n_chains = 4,
                        n_par_chains = 4,
                        iter_warmup = 1000,
                        iter_sampling = 1000
      )
      
    param_df <-  out_mod$param_df
    
    y_model_df <- param_df %>% dplyr::filter(complete_case ==FALSE)
      
      y_preds_full <- out_mod$y_pred
      y_preds <- y_preds_full$mean
      
      
     
      
    }
  }
  
  if(is.null(y_preds) ==FALSE){
    rmse_y <- sqrt(mean((y - y_preds)^2))
    
  }else{
    rmse_y <- NULL
  }
  

  
  comp_1 <- comp_fun(truth = x_u_1_true$mean,
                     fit = x_u_1_fit$mean,
                     exp = exp_x_u_1$mean,
                     var = "var 1",
                     index = index)
  
  comp_2 <- comp_fun(truth = x_u_2_true$mean,
                     fit = x_u_2_fit$mean,
                     exp = exp_x_u_2$mean,
                     var = "var 2",
                     index = index)
  
  rmse1 <- sqrt(mean(comp_1$diff^2))
  rmse2 <-  sqrt(mean(comp_2$diff^2))
  
  bias1 <- mean(comp_1$bias)
  bias2 <- mean(comp_2$bias)
  
  comb <- rbind(comp_1,comp_2)
  
  rmse_comb <-sqrt(mean(comb$diff^2))
  bias_comb <- mean(comb$bias)
  
  
  rmse_df <- data.frame(type = c("combined",
                                 "var 1",
                                 "var 2",
                                 "outcome"),
                        rmse = c(rmse_comb,
                                 rmse1,
                                 rmse2,
                                 rmse_y
                                 ),
                        bias = c(bias_comb,
                                 bias1,
                                 bias2,
                                 NA
                                 )) %>%
    dplyr::mutate(model_name = rep(mod_name,4),
                  fit_method = rep(fit_type,4),
                  index = rep(index,4))

    
  
  out <- list(comp_1 = comp_1,
              comp_2 = comp_2,
              comb = comb,
              rmse_df = rmse_df,
              mod_sum = vb_sum,
              x_u_1_fit_full = x_u_1_fit,
              x_u_2_fit_full = x_u_2_fit,
              x_u_1_fit = x_u_1_fit$mean,
              x_u_2_fit = x_u_2_fit$mean,
              x_u_1_true = x_u_1_true,
              x_u_2_true = x_u_2_true,
              exp_x_u_1 = exp_x_u_1,
              exp_x_u_2 = exp_x_u_2,
              y_preds = y_preds,
              y_preds_full = y_preds_full,
              rmse_y = rmse_y,
              y_model_df = y_model_df)
  
  out
}

