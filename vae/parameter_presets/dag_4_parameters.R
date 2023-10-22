# source files ------------------------

source(here::here("vae/parameter_presets/c_mat_function.R"))

rand_cmat <- FALSE
rand_beta <- TRUE
prob_cmat <- NULL

z_causes_x <- rep(1,3)
x_causes_x <- rep(0,3)
x_causes_r <- rep(1,3)
r_causes_r <- rep(1,3)
x_causes_y <- rep(1,3)

C_mat_function(z_causes_x = z_causes_x,
               x_causes_x = x_causes_x,
               x_causes_r = x_causes_r,
               r_causes_r = r_causes_r,
               x_causes_y = x_causes_y,
               n_z = n_z,
               n_x = n_x,
               n_r = n_r,
               source = TRUE,
               target = TRUE)





