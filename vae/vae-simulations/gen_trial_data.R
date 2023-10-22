# source in functions -------

source(here::here("vae-simulations/sim-full-data-funs.R"))

# Working parameters -------

n_z <- 3
n_x <- 3
n_r <- n_x
n <- 1000
mu_z <- 0
sd_z <- c(1,1,.5)
sd_x <- 1
sd_y <- 1

C_mat_x_z <- matrix(c(1,0,1,
                  0,1,1,
                  0,0,1), ncol = n_z, byrow = FALSE)

C_mat_x_x <- matrix(c(0,0,0,
                      1,0,0,
                      1,1,0), ncol = n_z, byrow = FALSE)


C_mat_r_x <- matrix(c(1,0,1,
                      0,1,1,
                      0,0,1), ncol = n_z, byrow = FALSE)

C_mat_r_r <-matrix(c(0,0,0,
                     1,0,0,
                     1,1,0), ncol = n_z, byrow = FALSE)
C_mat_y_x <- matrix(c(1,1,1), nrow = n_x, byrow = FALSE)

Beta_mat_x_z <- matrix(rnorm(n_z*n_x),nrow = n_z)
Beta_mat_x_x <- matrix(rnorm(n_x*n_x),nrow = n_x)
Beta_mat_r_x <- matrix(rnorm(n_r*n_x),nrow = n_r)
Beta_mat_r_r <- matrix(rnorm(n_r*n_r),nrow = n_r)
Beta_mat_y_x <- matrix(rnorm(1*n_r),nrow = n_x)

alpha_vec_x <- rep(0,n_x)
is_missing <- rep(FALSE,TRUE,TRUE)
r_prob_vec <- c(1,.9,.75)

trial <-  sim_full_once(n = n,
                        n_z = n_z,
                        mu_z = mu_z,
                        sd_z = sd_z,
                        n_x = n_x,
                        C_mat_x_z = C_mat_x_z,
                        Beta_mat_x_z = Beta_mat_x_z,
                        sd_x = sd_x,
                        C_mat_x_x = C_mat_x_x,
                        Beta_mat_x_x = Beta_mat_x_x,
                        x_link_fun = "identity",
                        alpha_vec_x = alpha_vec_x,
                        n_r = n_r,
                        C_mat_r_x =C_mat_r_x,
                        Beta_mat_r_x = Beta_mat_r_x,
                        C_mat_r_r =  C_mat_r_r,
                        Beta_mat_r_r = Beta_mat_r_r,
                        is_missing = is_missing,
                        r_link_fun = "expit",
                        C_mat_y_x = C_mat_y_x,
                        Beta_mat_y_x = Beta_mat_y_x,
                        sd_y = sd_y,
                        r_prob_vec = r_prob_vec
                        )


full_data <- trial$full_data
observed_data <- trial$obs_data

write.csv(full_data,"vae-simulations/sandbox/full_data.csv")
write.csv(observed_data,"vae-simulations/sandbox/observed_data.csv")

