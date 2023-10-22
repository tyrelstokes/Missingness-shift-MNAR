# Set some parameters for simulation --------

N <- 200

n_z <- 3
n_x <- 3
n_r <- n_x
n <- 1000
target_prop <- .2
mu_z <- 0
sd_z <- c(1,1,1)
sd_x <- 1
sd_y <- 1

alpha_vec_x <- rep(0,n_x)
is_missing <- rep(FALSE,TRUE,TRUE)
r_prob_vec <- c(1,.75,.7)
beta_sd <- 1.5

rand_cmat <- FALSE
rand_beta <- TRUE
prob_cmat <-NULL
C_mat_x_z = matrix(rep(1,9),nrow = n_z)
C_mat_x_x = matrix(rep(0,9),nrow = n_x)
C_mat_r_x = matrix(rep(1,9),nrow = n_x)
C_mat_r_r = matrix(rep(0,9),nrow = n_r)
C_mat_y_x = matrix(rep(1,3),nrow = n_x)

model_name_vec = c("Joint Model", "No outcome model", "No missingness model")
init = 0.2
iter = 20000
fit_method_vec <- c('vb')
n_chains = 4
n_par_chains = 4
iter_warmup = 1000
iter_sampling = 1000
use_rstan <- FALSE
index <- 1
pvae <- TRUE
uncertainty <- FALSE
n_draws <- 1000
tol <- .0001

Beta_mat_x_z <- matrix(rnorm(n_z*n_x, sd = beta_sd),nrow = n_z)
Beta_mat_x_x <- matrix(rnorm(n_x*n_x,sd = beta_sd),nrow = n_x)
Beta_mat_r_x <- matrix(rnorm(n_r*n_x,sd = beta_sd),nrow = n_r)
Beta_mat_r_r <- matrix(rnorm(n_r*n_r,sd = beta_sd),nrow = n_r)
Beta_mat_y_x <- matrix(rnorm(1*n_r,sd = beta_sd),nrow = n_x) 

Beta_mat_x_z_source <- Beta_mat_x_z
Beta_mat_x_x_source <- Beta_mat_x_x
Beta_mat_r_x_source <- Beta_mat_r_x
Beta_mat_r_r_source <- Beta_mat_r_r
Beta_mat_y_x_source <- Beta_mat_y_x

Beta_mat_x_z_target <- Beta_mat_x_z
Beta_mat_x_x_target <- Beta_mat_x_x
Beta_mat_r_x_target <- Beta_mat_r_x
Beta_mat_r_r_target <- Beta_mat_r_r
Beta_mat_y_x_target <- Beta_mat_y_x

C_mat_x_z_source <- C_mat_x_z 
C_mat_x_x_source <- C_mat_x_x 
C_mat_r_x_source <- C_mat_r_x 
C_mat_r_r_source <- C_mat_r_r 
C_mat_y_x_source <- C_mat_y_x 




Beta_mat_x_z_source <- Beta_mat_x_z
Beta_mat_x_x_source <- Beta_mat_x_x
Beta_mat_r_x_source <- Beta_mat_r_x
Beta_mat_r_r_source <- Beta_mat_r_r
Beta_mat_y_x_source <- Beta_mat_y_x



C_mat_x_z_target <- C_mat_x_z 
C_mat_x_x_target <- C_mat_x_x 
C_mat_r_x_target <- C_mat_r_x 
C_mat_r_r_target <- C_mat_r_r 
C_mat_y_x_target <- C_mat_y_x 
Beta_mat_x_z_target <- Beta_mat_x_z
Beta_mat_x_x_target <- Beta_mat_x_x
Beta_mat_r_x_target <- Beta_mat_r_x
Beta_mat_r_r_target <- Beta_mat_r_r
Beta_mat_y_x_target <- Beta_mat_y_x


x_link_fun <- "nl"
y_link_fun <- "identity"

n_layers <- 3
n_nodes <- rep(3,n_layers)

n_layers_x <- n_layers
n_layers_y <- n_layers

n_nodes_x <- n_nodes
n_nodes_y <- n_nodes

act_fun_y = "relu"
final_act_y = "linear"
act_fun_x = "relu"
final_act_x = "linear"
