# Housekeeping ---------------

`%>%` <- dplyr::`%>%`
`%do%` <- foreach::`%do%`

source(here::here("vae/vae-simulations/lin_pred_sim_functions.R"))
source(here::here("vae/vae-simulations/sim-expit-functions.R"))


### Missing Simulation ----------

r_simulation <- function(n,
                         n_r,
                         n_x,
                         x,
                         C_mat_x, #ncol = n_z, nrow = n_x,
                         Beta_mat_x, # nrow = n_x, ncol = n_z
                         C_mat_r,
                         Beta_mat_r,
                         is_missing,
                         link_function = "expit",
                         prob_vec
){
  
  
  lin_pred_list_xb <- lin_pred_fun(n_vars = n_r,
                                   parent_var = x,
                                   C_mat = C_mat_x,
                                   Beta_mat = Beta_mat_x)
  
  
  if(link_function == "expit"){
    out <- gen_expit_link(lin_pred_list = lin_pred_list_xb,
                          n_var = n_r,
                          n = n,
                          var_name = "r",
                          C_mat_r = C_mat_r,
                          Beta_mat_r = Beta_mat_r,
                          prob_vec = prob_vec)
    
  }
  out
}

