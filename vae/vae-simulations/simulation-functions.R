# Housekeeping ---------

`%>%` <- dplyr::`%>%`
`%do%` <- foreach::`%do%`

# Working parameters -------

n_z <- 3
n_x <- 3
n <- 1000
mu_z <- 0
sd_z <- c(1,1,.5)

C_mat <- matrix(c(1,0,1,
                  0,1,1,
                  0,0,1), ncol = n_z, byrow = TRUE)

Beta_mat <- matrix(rnorm(n_z*n_x),nrow = n_z)


# single lin pred function ------------

lin_pred_once <- function(parent_var,
                          C_vec,
                          Beta_vec){
  
  w_cols <- (C_vec ==1)
  
  if((length(w_cols)>0) &(sum(w_cols) >0)){
  w_eff <- parent_var[,w_cols]
  b_eff <- Beta_vec[w_cols]
  
  lin_pred <- if(length(b_eff) >1){
    w_eff %*% b_eff
  }else{
    w_eff*b_eff
    
  }
  }else{
    lin_pred <- 0
  }
  lin_pred
}

# Linear Predictor function ------------------


lin_pred_fun <- function(n_vars,
                         parent_var,
                         C_mat,
                         Beta_mat){

lin_pred_list <- lapply(c(1:n_vars),function(j){
  
 lin_pred <-  lin_pred_once(parent_var = parent_var,
                C_vec = C_mat[j,],
                Beta_vec = Beta_mat[j,])
  
  lin_pred
})

lin_pred_list

}

# identity link outcome generation -----------

gen_identity_link <- function(lin_pred_list,
                              n_var,
                              sd_vec,
                              var_name,
                              mean_vec = 0){
  
  if((length(mean_vec) ==1) &n_var >1){
    mean_vec <- rep(mean_vec,n_var)
  }
  
  mu <- matrix(unlist(lin_pred_list), ncol = n_var)
  x <- matrix(unlist(lapply(c(1:n_var),function(i){
    xi <- rnorm(n,mu[,i],sd = sd_vec[i]) + mean_vec[i]
    xi
  })),ncol = n_var)
  
  out <- list(x,
              mu)
  
  names(out) <- c(var_name,paste0("mu_",var_name))
  
  out
}


# Alpha finder -----------------------------------------

expit_fun <- function(alpha,
                      xb){
  locfit::expit(alpha + xb)
}

l2_expit <- function(x,
                     xb,
                     prob){
  
  (prob - mean(expit_fun(alpha = x,
                    xb = xb)))^2
  
}

find_alpha <- function(xb,
                       prob,
                       link_function ="expit"){
  
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
 } 
 
  
  
  out <- list(alpha = alpha,
              expit = expit)
  
  out
}



# generate expit link ---------------------

gen_expit_link_once <- function(xb,
                                prob_r,
                                rb,
                                n){

  
  xb <- xb
  
  lin_pred <- xb + rb
  
  alpha_exp_list <- find_alpha(xb = lin_pred,
                               prob = prob_r,
                               link_function = "expit")
  
  r <- rbinom(n,
              size = 1,
              prob = alpha_exp_list$expit)
  
alpha_exp_list$r <- r

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

# latent variable simulation --------------

z_simulation <- function(n,
                         n_z,
                         mu_z,
                         sd_z){
  if((length(mu_z)==1) &(n_z >1)){
    mu_z <- rep(mu_z,n_z)
  }
  if((length(sd_z)==1)&(n_z >1)){
    sd_z <- rep(sd_z,n_z)
  }
  
  z <- matrix(nrow =n, ncol = n_z,unlist(lapply(c(1:n_z),function(i){
    x <- rnorm(n,mu_z[i],sd_z[i])
    x
  })))
  
  z
  
}

# Test z_simulation ------------

z <- z_simulation(n=n,
             n_z = n_z,
             mu_z = mu_z,
             sd_z = sd_z
             )

# X Simulation -----------------

x_simulation <- function(n,
                         n_x,
                         z,
                         C_mat_z, #ncol = n_z, nrow = n_x,
                         Beta_mat_z, # nrow = n_x, ncol = n_z
                         sd_x,
                         C_mat_x,
                         Beta_mat_x,
                         link_function = "identity"
){
  
  
  
  lin_pred_list_zb <- lin_pred_fun(n_vars = n_x,
                           parent_var = z,
                           C_mat = C_mat_z,
                           Beta_mat = Beta_mat_z)
  
  xb <- matrix(nrow = n, ncol = n_x)
  for(i in 1:n_x){
    
    if(i ==1){
      xb[,i] <- rep(0,n)
    }else{
      ind <- (1:(i-1))
      x_prev <- x[,ind]
      c_prev <- C_mat_x[ind,i]
      b_prev <- Beta_mat_x[ind,i]
      
      
   xb[,i] <-  lin_pred_once(parent_var = x_prev,
                  C_vec = c_prev,
                  Beta_vec = b_prev)
    
    
      
    }
  }
  
  lin_pred_comb <- lapply(c(1:n_x),function(i){
    lin_pred_list_zb[[i]] + xb[,i]
  })

 
 if((length(sd_x)==1) &(n_x >1)){
   sd_x <- rep(sd_x,n_x)
 }
 
  
 if(link_function =="identity"){
   
   out <- gen_identity_link(lin_pred_list = lin_pred_comb,
                            n_var = n_x,
                            sd_vec = sd_x,
                            var_name = "x")
   
 }
   
 out
 
}


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
                         link_function = "expit"
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
  

# Outcome Simulation ---------------

y_simulation <- function(n_y,
                         x,
                         C_mat,
                         Beta_mat,
                         sd_y,
                         link_function = "identity"){
  
  
  lin_pred_list <- lin_pred_fun(n_vars = 1,
                                parent_var = x,
                                C_mat = C_mat,
                                Beta_mat = Beta_mat)
  
  
  if(link_function =="identity"){
    
    out <- gen_identity_link(lin_pred_list = lin_pred_list,
                             n_var = 1,
                             sd_vec = sd_y,
                             var_name = "y")
    
  } 
  
  
}

