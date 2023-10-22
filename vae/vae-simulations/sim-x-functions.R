# Housekeeping ---------------

`%>%` <- dplyr::`%>%`
`%do%` <- foreach::`%do%`

source(here::here("vae/vae-simulations/lin_pred_sim_functions.R"))
source(here::here("vae/vae-simulations/sim-identity-link-functions.R"))
source(here::here("vae/vae-simulations/simulate-nl-layers.R"))

# X Simulation -----------------

x_simulation <- function(n,
                         n_x,
                         z,
                         C_mat_z, #ncol = n_z, nrow = n_x,
                         Beta_mat_z, # nrow = n_x, ncol = n_z
                         sd_x,
                         C_mat_x,
                         Beta_mat_x,
                         link_function = "identity",
                         alpha_vec,
                         n_nodes = c(2,2),
                         n_layers = 2,
                         act_fun = "relu",
                         final_act_fun = "linear"
){
  
  xb <- matrix(nrow = n, ncol = n_x)
  x <- matrix(nrow = n, ncol = n_x)
  total_lin_pred <- matrix(nrow = n, ncol = n_x)
  
  if((length(sd_x)==1) &(n_x >1)){
    sd_x <- rep(sd_x,n_x)
  }
  
  if(link_function == "identity"){
  lin_pred_list_zb <- lin_pred_fun(n_vars = n_x,
                                   parent_var = z,
                                   C_mat = C_mat_z,
                                   Beta_mat = Beta_mat_z)
  

  

  for(i in 1:n_x){
    
    if(i ==1){
      xb[,i] <- rep(0,n)
      x[,i] <- gen_identity_link_once(lin_pred = lin_pred_list_zb[[i]],
                                      std = sd_x[i],
                                      alpha = alpha_vec[i],
                                      n = n)
    }else{
      ind <- (1:(i-1))
      x_prev <- x[,ind]
      c_prev <- C_mat_x[ind,i]
      b_prev <- Beta_mat_x[ind,i]
      
      
      xb[,i] <-  lin_pred_once(parent_var = x_prev,
                               C_vec = c_prev,
                               Beta_vec = b_prev)
      
      total_lin_pred[,i] = alpha_vec[i] + (lin_pred_list_zb[[i]]+xb[,i])
      
      x[,i] <- gen_identity_link_once(lin_pred = (lin_pred_list_zb[[i]]+xb[,i]),
                                      std = sd_x[i],
                                      alpha = alpha_vec[i],
                                      n = n)
      
    }
  }
  
  lin_pred_mat <- matrix(nrow = n,unlist(lapply(c(1:n_x),function(i){
    lin_pred_list_zb[[i]] + xb[,i] + alpha_vec[i]
  })))
  
  
  resid_mat <- x - total_lin_pred
  
  total_var <- apply(x,2,var)
  resid_var <- apply(resid_mat,2,var)
  
  }else{
    
    if(link_function == "nl"){
   z_nl <-   sim_many_layers(parent_var = z,
                      C_mat = C_mat_z,
                      n_vars = n_z,
                      act_fun = act_fun,
                      n_nodes = n_nodes,
                      n_layers = n_layers,
                      final_output_act_fun = final_act_fun,
                      n = n)
   
   for(i in 1:n_x){
     if(i ==1){
       xb[,i] <- z_nl$value_list[[i]]
       total_lin_pred[,i] = alpha_vec[i] + xb[,i]
     x[,i] <- gen_identity_link_once(lin_pred = total_lin_pred[,i],
                                     std = sd_x[i],
                                     alpha = alpha_vec[i],
                                     n = n)
     
     }else{
       nv <- (i-1)
       
       c_mat_eff <- as.matrix(C_mat_x[1:(i-1),i])
       x_eff <- as.matrix(x[,1:(i-1)])
       
       if(sum(c_mat_eff == 0)==(nrow(c_mat_eff)*ncol(c_mat_eff))){
         xb[,i] <- rep(0,nrow(x_eff))
       }else{
         
         xb_list <-  sim_many_layers(parent_var = x_eff,
                                     C_mat = c_mat_eff,
                                     n_vars = nv,
                                     act_fun = act_fun,
                                     n_nodes = n_nodes,
                                     n_layers = n_layers,
                                     final_output_act_fun = final_act_fun,
                                     n = n)
         
    xb[,i] <- xb_list$value_list[[1]] 
       }
    
    total_lin_pred[,i] = alpha_vec[i] + xb[,i] +z_nl$value_list[[i]]
    x[,i] <- gen_identity_link_once(lin_pred = total_lin_pred[,i],
                                    std = sd_x[i],
                                    alpha = alpha_vec[i],
                                    n = n)
    
       
     }
     
   }
      
    }
    
    lin_pred_mat <- matrix(nrow = n,unlist(lapply(c(1:n_x),function(i){
      z_nl$value_list[[i]] + xb[,i] + alpha_vec[i]
    })))
    
    
    resid_mat <- x - total_lin_pred
    
    total_var <- apply(x,2,var)
    resid_var <- apply(resid_mat,2,var)
    
    
    
  }
  
  emp_sd_df <- data.frame(var = c("x1","x2","x3"),
                          total_var = total_var,
                          resid_var = resid_var)
  
 out <- list(lin_pred = lin_pred_mat,
             mu_x = lin_pred_mat + alpha_vec,
             x = x,
             total_lin_pred = total_lin_pred,
             emp_sd_df = emp_sd_df)
  
  out
  
}
