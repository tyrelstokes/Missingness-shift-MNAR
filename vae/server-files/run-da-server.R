#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

# source files

source(here::here("vae/parameter_presets/c_mat_function.R"))
source(here::here("vae/parameter_presets/select_dag_fun.R"))
source(here::here("vae/parameter_presets/beta_gen_fun.R"))
source(here::here("vae/vae-simulations/simulate-and-run-domain-adaptation.R"))

# Set some parameters for simulation --------

N <- as.numeric(args[1])

n <- as.numeric(args[2])

n_z <- 3
n_x <- 3
n_r <- n_x
target_prop <- as.numeric(args[3])
mu_z <- 0
sd_z <- c(1,1,1)
sd_x <- 1
sd_y <- 1

alpha_vec_x <- rep(0,n_x)
is_missing <- rep(FALSE,TRUE,TRUE)
r_prob_vec <- c(1,as.numeric(args[4]),as.numeric(args[5]))
beta_sd <- 1.5

which_dag <- as.numeric(args[6])

select_dag_function(which_dag)


model_name_vec = c("Z model","Joint Model", "No outcome model", "No missingness model")
init = 0.2
iter = 20000
fit_method_vec <- c('vb')
n_chains = 4
n_par_chains = 4
iter_warmup = 1000
iter_sampling = 1000
use_rstan <- FALSE
index <- 1
pvae <- TRUE
uncertainty <- FALSE
n_draws <- 1000
tol <- .0001

beta_gen_function(beta_sd)


x_link_fun <- "nl"
y_link_fun <- "identity"

n_layers <- as.numeric(args[7])
n_nodes <- rep(3,n_layers)

n_layers_x <- n_layers
n_layers_y <- n_layers

n_nodes_x <- n_nodes
n_nodes_y <- n_nodes

act_fun_y = "relu"
final_act_y = "linear"
act_fun_x = "relu"
final_act_x = "linear"

is_server = TRUE

if(is_server == TRUE){
  python_int <- "python"
}else{
  python_int <- "vae/env/bin/python"
}

###########################################
#first_run <- TRUE

#if(first_run ==TRUE){
 # file.remove(here::here("vae/impute_models/joint_impute_model_3"))
#  file.remove(here::here("vae/impute_models/z_model"))
 # file.remove(here::here("vae/impute_models/joint_no_missing_model"))
#  file.remove(here::here("vae/impute_models/joint_no_outcome"))
 # file.remove(here::here("vae/outcome-models/y-model"))
  
#}


############################################
# Run the simulations 

try <- sim_and_run_da_many(N = N,
                           n_z = n_z,
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
                           prob_cmat = 0.7,
                           rand_beta = rand_beta,
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



rmse_y_df <- try$rmse_df_y
rmse_x_df <- try$rmse_x_df

rmse_y_df$model_name <- rmse_y_df$model

meta_df <- data.frame(N = N,
                      n = n,
                      target_prop = target_prop,
                      miss_prob_2 = 1 - r_prob_vec[2],
                      miss_prob_3 = 1 - r_prob_vec[3],
                      dag = which_dag,
                      x_link_fun = x_link_fun,
                      y_link_fun = y_link_fun,
                      beta_sd = beta_sd,
                      n_nodes_x = paste(as.character(n_nodes_x),collapse = ","),
                      n_layers_x = n_layers_x,
                      n_nodes_y = paste(as.character(n_nodes_y),collapse = ","),
                      n_layers_y = n_layers_y,
                      act_fun_y = act_fun_y,
                      final_act_y = final_act_y,
                      act_fun_x = act_fun_x,
                      final_act_x = final_act_x)






# Source the plotting functions--------------

source(here::here("vae/results-organizing/plot-summary-funs.R"))

rmse_y_df <- merge_meta_rmse(meta_df = meta_df,
                             rmse_frame = rmse_y_df)

rmse_x_df <- merge_meta_rmse(meta_df = meta_df,
                             rmse_frame = rmse_x_df)



# save data frames

p_number <- save_data(dt = rmse_y_df,
                      fname = "rmse_y",
                      findit = TRUE,
                      num = 0,
                      ext = ".csv")

save_data(dt = rmse_x_df,
          fname = "rmse_x",
          findit = FALSE,
          num = p_number,
          ext = ".csv")

save_data(dt = meta_df,
          fname = "meta_df",
          findit = FALSE,
          num = p_number,
          ext = ".csv")

meta_list <- list(meta_df,
                  Beta_mat_x_z_source = Beta_mat_x_z_source,
                  Beta_mat_x_x_source = Beta_mat_x_x_source,
                  Beta_mat_r_x_source = Beta_mat_r_x_source,
                  Beta_mat_r_r_source = Beta_mat_r_r_source,
                  Beta_mat_y_x_source = Beta_mat_y_x_source)

save_data(dt = meta_list,
          fname = "meta_list",
          findit = FALSE,
          num = p_number,
          ext = ".Rda")


