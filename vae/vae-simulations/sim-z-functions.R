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
