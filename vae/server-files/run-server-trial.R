#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

# Source the files

source(here::here("vae/impute_models/stan-model-run-functions.R"))
source(here::here("vae/results-organizing/plot-summary-funs.R"))

# Set some parameters ---------

suff <- args[1]

N <- as.numeric(args[2])

n <- as.numeric(args[3])

n_z <- 3
n_x <- 3
n_r <- n_x

mu_z <- 0
sd_z <- c(1,1,1)
sd_x <- 1
sd_y <- 1

alpha_vec_x <- rep(0,n_x)
is_missing <- rep(FALSE,TRUE,TRUE)
r_prob_vec <- c(1,.75,.7)
beta_sd <- 1.5

rand_cmat <- FALSE
rand_beta <- TRUE
prob_cmat <-NULL
C_mat_x_z = matrix(rep(1,9),nrow = n_z)
C_mat_x_x = matrix(rep(0,9),nrow = n_x)
C_mat_r_x = matrix(rep(1,9),nrow = n_x)
C_mat_r_r = matrix(rep(0,9),nrow = n_r)
C_mat_y_x = matrix(rep(1,3),nrow = n_x)
Beta_mat_x_z =  NULL
Beta_mat_x_x =  NULL
Beta_mat_r_x =  NULL
Beta_mat_r_r =  NULL
Beta_mat_y_x =  NULL
model_name_vec = c("Z model",
                   "Joint Model",
                   "No outcome model",
                   "No missingness model")
init = 0.2
iter = 20000
fit_method_vec <- c('vb')
n_chains = 4
n_par_chains = 4
iter_warmup = 1000
iter_sampling = 1000
use_rstan <- args[4]

# Simulate ------------------

reff <- sim_and_run_many(N = N,
                         n_z = n_z,
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
                         use_rstan = use_rstan)

rmse_frame <- reff$rmse_df

#sum_df <-summary_fun(rmse_frame = rmse_frame)

#print(sum_df)

p1 <- rmse_plot(rmse_frame = rmse_frame)

p2 <- bias_plot(rmse_frame = rmse_frame)

frame_pref <- "server-files/data-frames/rmse_frame_"
plot_pref <- "server-files/plots/"

write.csv(rmse_frame, paste0("server-files/data-frames/rmse_frame_",paste0(suff,".csv")))
#write.csv(sum_df, paste0("server-files/data-frames/summary_df_",paste0(suff,".csv")))

ggplot2::ggsave(filename = paste0(plot_pref,
              paste0("rmse-plot_",paste0(suff,".png"))),
              plot = p1)

ggplot2::ggsave(filename = paste0(plot_pref,
                                  paste0("bias-plot_",paste0(suff,".png"))),
                plot = p2)
          
          
          