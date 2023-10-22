# source functions

source(here::here("vae/impute_models/run_python_functions.R"))


# mice function

mice_function <- function(observed_data,
                          full_data,
                          data_type,
                          index = 1,
                          method = "pmm"){
  
  
  mice_object <- mice::mice(observed_data[,c("xobs_1","xobs_2","xobs_3","y")],
                            m = 1,
                            meth = method,
                            ri.maxit = 5)
  
  completed_data <- mice::complete(mice_object)
  
  completed_small <- completed_data[,c(1:3)]
  
  m_name <- paste("mice", method)
  
  rmse_list <- rmse_calc(observed_data = observed_data,
                         full_data = full_data,
                         imputed_data = completed_small,
                         model_name = m_name,
                         data_type = data_type,
                         index = index)
  
  x_u_imputed <- cbind(completed_data$xobs_2,
                       completed_data$xobs_3)
  
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
                                rmse_list)
  
  
  
  
  out
  
  
  
}



