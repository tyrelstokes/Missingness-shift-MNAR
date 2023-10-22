# Housekeeping

`%>%` <- dplyr::`%>%`
source(here::here("vae/outcome-models/model-utility-functions.R"))
# Run outcome models----------------

fit_fun <- function(stan_mod,
                    fit_method,
                    stan_list,
                    init = 0.2,
                    iter,
                    n_chains = 4,
                    n_par_chains = 4,
                    iter_warmup = 1000,
                    iter_sampling = 1000){
  
  y_mod <- cmdstanr::cmdstan_model("vae/outcome-models/y-model.stan")
  
  if(fit_method =="vb"){
    fit <- y_mod$variational(data = stan_list,
                             init = init)
  }
  
  if(fit_method =="mcmc"){
    fit <- y_mod$sample(data = stan_list,
                       init = init,
                       chains = n_chains,
                       parallel_chains = n_par_chains,
                       iter_warmup = iter_warmup,
                       iter_sampling = iter_sampling
                       )
  }
  
  if(fit_method == "opt"){
    fit <- y_mod$optimize(data = stan_list,
                          init = init)
  }
  
  fit
  
}

# get row ---------------------

get_quant <- function(vb_summer,
                      nm){
  
  out <- vb_summer[grepl(nm,vb_summer$variable),]
  out
}

# get param df ------------

get_param_df <- function(vb_summer){
  
  alpha <- get_quant(vb_summer = vb_summer,
                     nm = "alpha")
  
  beta_o <- get_quant(vb_summer = vb_summer,
                      nm = "beta_o")
  
  beta_u <- get_quant(vb_summer = vb_summer,
                      nm = "beta_u")
  
  sigma <- get_quant(vb_summer = vb_summer,
                     nm = "sigma")
  
  param_df <- rbind(alpha,
                    beta_o,
                    beta_u,
                    sigma)
  
  param_df$complete_case <- grepl("cc",param_df$variable)
  param_df <- param_df %>% dplyr::arrange(complete_case)
  
  param_df
}


# Get preds -------------------------------

get_preds <- function(vb_summer){
  
  pred <- get_quant(vb_summer = vb_summer,
                     nm = "pred\\[")
  
  pred_cc <- get_quant(vb_summer = vb_summer,
                       nm = "pred_cc")
  
  out <- list(pred = pred,
              pred_cc = pred_cc)
  
  out
  
}

# Extract quantities from fit model ------------

extract_outcome <- function(fitted,
                            y){
  
  vb_summer <- fitted$summary()
  
  param_df <- get_param_df(vb_summer = vb_summer)

  preds <- get_preds(vb_summer = vb_summer)
  
  y_pred <- preds$pred
  y_pred_cc <- preds$pred_cc
  
  rmse_1 <- sqrt(mean((y - y_pred$mean)^2))
  rmse_2 <- sqrt(mean((y - y_pred_cc$mean)^2))
  
  rmse_df <- data.frame(rmse = c(rmse_1,rmse_2),
                        type = c("Full", "Complete Case"))
  
  out <- list(vb_summer = vb_summer,
              param_df = param_df,
              y_pred = y_pred,
              y_pred_cc = y_pred_cc,
              rmse_df)
  
  out
  
}


# Run outcome function --------

run_outcome_model <- function(y,
                              x_u,
                              x_o,
                              R,
                              fit_method,
                              init,
                              iter,
                              n_chains = 4,
                              n_par_chains = 4,
                              iter_warmup = 1000,
                              iter_sampling = 1000
                              ){
  
  stan_lister <- list(y = y,
                      x_u = x_u,
                      x_o = x_o,
                      N = length(y),
                      R = R)
  
  y_mod <- cmdstanr::cmdstan_model(here::here("vae/outcome-models/y-model.stan"))
  
 
 fitted <-    fit_fun(stan_mod = y_mod,
                      fit_method = fit_method,
                      stan_list = stan_lister,
                      init = init,
                      iter = iter,
                      n_chains = n_chains,
                      n_par_chains = n_par_chains,
                      iter_warmup = iter_warmup,
                      iter_sampling = iter_sampling)
 
 model_quantities <- extract_outcome(fitted = fitted,
                                     y = y)
 
 
 model_quantities
  
}
