# Housekeeping ---------

`%>%` <- dplyr::`%>%`
`%do%` <- foreach::`%do%`

source(here::here("vae/vae-simulations/lin_pred_sim_functions.R"))


# Expit Function --------------
expit_fun <- function(alpha,
                      xb){
  locfit::expit(alpha + xb)
}

# L2 loss function for finding alpha ---------------
l2_expit <- function(x,
                     xb,
                     prob){
  
  if(prob != 1){
  (prob - mean(expit_fun(alpha = x,
                         xb = xb)))^2
    
  }
  
}

# Alpha finder -----------------------------------------
find_alpha <- function(xb,
                       prob,
                       link_function ="expit"){
  
  if(prob!= 1){
  if(link_function =="expit"){
    
    opt <-   optim(par = 0,
                   fn = l2_expit,
                   method = "Brent",
                   xb = xb,
                   prob = prob,
                   lower = -40,
                   upper = 40)
    
    alpha <- opt$par
    
    expit <- expit_fun(alpha = alpha,
                       xb = xb)
    
    out <- list(alpha = alpha,
                expit = expit)
    
  } 
  
  }else{
    
    
    out <- list(alpha = Inf,
                expit = rep(1,length(xb)))
  }
  
  

  out
}



# generate expit link ---------------------

gen_expit_link_once <- function(xb,
                                prob_r,
                                rb,
                                n,
                                is_missing = FALSE){
  
  if(is_missing == FALSE){
  xb <- xb
  
  lin_pred <- xb + rb
  
  alpha_exp_list <- find_alpha(xb = lin_pred,
                               prob = prob_r,
                               link_function = "expit")
  
  r <- rbinom(n,
              size = 1,
              prob = alpha_exp_list$expit)
  
  alpha_exp_list$r <- r
  }else{
    alpha_exp_list <- list(alpha = Inf,
                           expit = rep(Inf,n),
                           r = rep(1,n))
  }
  
  alpha_exp_list
  
  
}




gen_expit_link <- function(lin_pred_list,
                           n_var,
                           n,
                           var_name,
                           C_mat_r, # n_r x n_x
                           Beta_mat_r, #n_r x n_x,
                           prob_vec){
  
  
  alpha <- vector(length = n_var)
  r <- matrix(nrow = n, ncol = n_var)
  expit <- matrix(nrow = n, ncol = n_var)
  rb <- matrix(nrow = n,ncol = n_var)
  
  for(i in 1:n_var){
    
    if(i < 2){
      rb[,i] <- rep(0,n)
      
      out_list <-  gen_expit_link_once(xb = lin_pred_list[[i]],
                                       prob_r = prob_vec[i],
                                       rb = rb[,i],
                                       n = n)
      alpha[i] <- out_list$alpha
      r[,i] <- out_list$r
      expit[,i] <- out_list$expit
      
    }else{
      ind <- (1:(i-1))
      r_prev <- r[,ind]
      c_prev <- C_mat_r[ind,i]
      b_prev <- Beta_mat_r[ind,i]
      
      rb[,i] <-  lin_pred_once(parent_var = r_prev,
                               C_vec = c_prev,
                               Beta_vec = b_prev)
      
      out_list <-    gen_expit_link_once(xb = lin_pred_list[[i]],
                                         prob_r = prob_vec[i],
                                         rb = rb[,i],
                                         n = n)
      
      alpha[i] <- out_list$alpha
      r[,i] <- out_list$r
      expit[,i] <- out_list$expit
      
    }
    
    out_list <-    gen_expit_link_once(xb = lin_pred_list[[i]],
                                       prob_r = prob_vec[i],
                                       rb = rb[,i],
                                       n = n)
    
    
    alpha[i] <- out_list$alpha
    r[,i] <- out_list$r
    expit[,i] <- out_list$expit
    
  }
  
  
  
  out <- list(alpha = alpha,
              r = r,
              mu= expit,
              rb = rb)
  
}

