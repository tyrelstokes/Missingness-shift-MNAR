source(here::here("vae/parameter_presets/c_mat_function.R"))

beta_gen_function <- function(beta_sd){

    Beta_mat_x_z <- matrix(rnorm(n_z*n_x, sd = beta_sd),nrow = n_z)
    Beta_mat_x_x <- matrix(rnorm(n_x*n_x,sd = beta_sd),nrow = n_x)
    Beta_mat_r_x <- matrix(rnorm(n_r*n_x,sd = beta_sd),nrow = n_r)
    Beta_mat_r_r <- matrix(rnorm(n_r*n_r,sd = beta_sd),nrow = n_r)
    Beta_mat_y_x <- matrix(rnorm(1*n_r,sd = beta_sd),nrow = n_x) 
    
    
    assign_function(Beta_mat_x_z,source = TRUE, target = TRUE)
    assign_function(Beta_mat_x_x,source = TRUE, target = TRUE)
    assign_function(Beta_mat_r_x,source = TRUE, target = TRUE)
    assign_function(Beta_mat_r_r,source = TRUE, target = TRUE)
    assign_function(Beta_mat_y_x,source = TRUE, target = TRUE)
    
    


}