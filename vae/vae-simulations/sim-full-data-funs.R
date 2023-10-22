# Housekeeping ---------------

`%>%` <- dplyr::`%>%`
`%do%` <- foreach::`%do%`

source(here::here("vae/vae-simulations/lin_pred_sim_functions.R"))
source(here::here("vae/vae-simulations/sim-identity-link-functions.R"))
source(here::here("vae/vae-simulations/sim-z-functions.R"))
source(here::here("vae/vae-simulations/sim-x-functions.R"))
source(here::here("vae/vae-simulations/sim-r-functions.R"))
source(here::here("vae/vae-simulations/sim-y-functions.R"))
source(here::here("vae/vae-simulations/simulate-nl-layers.R"))

# sim data set once

sim_full_once <- function(n,
                          n_z,
                          mu_z,
                          sd_z,
                          n_x,
                          C_mat_x_z,
                          Beta_mat_x_z,
                          sd_x,
                          C_mat_x_x,
                          Beta_mat_x_x,
                          x_link_fun = "identity",
                          alpha_vec_x,
                          n_r,
                          C_mat_r_x,
                          Beta_mat_r_x,
                          C_mat_r_r,
                          Beta_mat_r_r,
                          is_missing,
                          r_link_fun = "expit",
                          C_mat_y_x,
                          Beta_mat_y_x,
                          sd_y,
                          r_prob_vec,
                          alpha_y = 0,
                          y_link_fun = "identity",
                          n_nodes_x = c(2,2),
                          n_layers_x = 2,
                          n_nodes_y = c(2,2),
                          n_layers_y = 2,
                          act_fun_y = "relu",
                          final_act_y = "linear",
                          act_fun_x = "relu",
                          final_act_x = "linear"){
  
  z <- z_simulation(n = n,
                    n_z = n_z,
                    mu_z = mu_z,
                    sd_z = sd_z)
  
x_list <- x_simulation(n = n,
                  n_x = n_x,
                  z = z,
                  C_mat_z = C_mat_x_z, #ncol = n_z, nrow = n_x,
                  Beta_mat_z = Beta_mat_x_z, # nrow = n_x, ncol = n_z
                  sd_x = sd_x,
                  C_mat_x =  C_mat_x_x,
                  Beta_mat_x = Beta_mat_x_x,
                  link_function = x_link_fun,
                  alpha_vec = alpha_vec_x,
                  n_nodes = n_nodes_x,
                  n_layers = n_layers_x,
                  act_fun = act_fun_x,
                  final_act_fun = final_act_x)

x <- x_list$x

x_lin_pred <- x_list$total_lin_pred

emp_sd_vec_x <- x_list$emp_sd_df

r_list <- r_simulation(n = n,
                  n_r = n_r,
                  n_x = n_x,
                  x = x,
                  C_mat_x = C_mat_r_x, #ncol = n_z, nrow = n_x,
                  Beta_mat_x = Beta_mat_r_x, # nrow = n_x, ncol = n_z
                  C_mat_r = C_mat_r_r,
                  Beta_mat_r = Beta_mat_r_r,
                  is_missing = is_missing,
                  link_function = r_link_fun ,
                  prob_vec = r_prob_vec
)

r <- r_list$r

y_list <- y_simulation(n_y = 1,
                       x = x,
                       C_mat = C_mat_y_x,
                       Beta_mat = Beta_mat_y_x,
                       sd_y = sd_y,
                       link_function = y_link_fun,
                       alpha_y = alpha_y,
                       n = n,
                       n_nodes = n_nodes_y,
                       n_layers = n_layers_y,
                       activation_function = act_fun_y,
                       final_output_act_fun = final_act_y)

y <- y_list

full_data <- data.frame(z,x,r,y)
names(full_data) <- c(paste0("z_",c(1:n_z)),
                      paste0("x_",c(1:n_x)),
                      paste0("r_",c(1:n_r)),
                      "y")

lin_mod <- lm(y ~ x_1 + x_2 + x_3, data = full_data)
full_data_lin_coefs_est <- coef(lin_mod)
est_sd_y <- sigma(lin_mod)
full_data_lin_coefs_est <- c(full_data_lin_coefs_est,est_sd_y)
names(full_data_lin_coefs_est) <- c("alpha_y","x_1","x_2","x_3","sigma_y")

lin_y_x_coefs <- Beta_mat_y_x*C_mat_y_x

linear_coefs <- data.frame(alpha_y = alpha_y,
                           b1 = lin_y_x_coefs[1],
                           b2 = lin_y_x_coefs[2],
                           b3 = lin_y_x_coefs[3],
                           sd_y = sd_y)

x_obs <- matrix(nrow = n,ncol = n_x,
                unlist(
                   lapply(c(1:n_x), function(i){
                     ifelse(r[,i] ==0,NA,x[,i])
                   })
                )
)

obs_data <- data.frame(x_obs,r,y)
names(obs_data) <- c(paste0("xobs_",c(1:n_x)),
                     paste0("r_",c(1:n_r)),
                     "y")
                
out <- list(full_data = full_data,
            obs_data = obs_data,
            x_lin_pred = x_lin_pred,
            emp_sd_vec_x = emp_sd_vec_x,
            full_data_lin_coefs_est = full_data_lin_coefs_est,
            linear_coefs = linear_coefs)

out
}








