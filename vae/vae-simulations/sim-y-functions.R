# Housekeeping ---------------

`%>%` <- dplyr::`%>%`
`%do%` <- foreach::`%do%`

source(here::here("vae/vae-simulations/lin_pred_sim_functions.R"))
source(here::here("vae/vae-simulations/sim-identity-link-functions.R"))


# Outcome Simulation ---------------

y_simulation <- function(n_y,
                         x,
                         C_mat,
                         Beta_mat,
                         sd_y,
                         link_function = "identity",
                         alpha_y,
                         n,
                         n_nodes = c(2,2),
                         n_layers = 2,
                         activation_function = "relu",
                         final_output_act_fun = "linear"){
  
  
  
  
  
  if(link_function =="identity"){
  
  lin_pred <- lin_pred_once(parent_var = x,
                            C_vec = C_mat,
                            Beta_vec = Beta_mat)
 
    out <- gen_identity_link_once(lin_pred = lin_pred,
                                  std = sd_y,
                                  alpha = alpha_y,
                                  n = n)

  }
  
  if(link_function =="nl"){
    y_list <-   sim_many_layers(parent_var = as.matrix(x),
                                C_mat = as.matrix(C_mat),
                                n_vars = 1,
                                act_fun = activation_function,
                                n_nodes = n_nodes,
                                n_layers = n_layers,
                                final_output_act_fun = final_output_act_fun,
                                n = n)
    
    out <- unlist(y_list$value_list)
  }
  
  
  
  
  out
}
