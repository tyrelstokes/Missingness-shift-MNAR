# Housekeeping ---------

`%>%` <- dplyr::`%>%`
`%do%` <- foreach::`%do%`


# identity link once ---------------------

gen_identity_link_once <- function(lin_pred,
                                   std,
                                   alpha,
                                   n){
  mu <- lin_pred + alpha
  outcome <- rnorm(n, mean = mu, sd = std)
  
  outcome
}


# identity link outcome generation -----------

gen_identity_link <- function(lin_pred_list,
                              n_var,
                              sd_vec,
                              var_name,
                              mean_vec = 0,
                              n){
  
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
