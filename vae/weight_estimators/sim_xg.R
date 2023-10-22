n <- 500
p <- 3

x <- matrix(rnorm(n*p),ncol = p)

simple_simulation <- function(p_cont,
                              p_bin,
                              n,
                              alpha = rnorm(1),
                              beta = rnorm((p_cont+p_bin))){
  
  
  x_cont <- matrix(rnorm(n*p_cont),ncol = p_cont) 
  
  x_bin <- matrix(rbinom(n*p_bin,size = 1, prob = .5), ncol = p_bin)
  
  x <- cbind(x_cont,x_bin)
  
  

  
  mu <- locfit::expit(alpha + x%*%beta)
  
  y <- rbinom(n, size = 1, prob = mu)
  y <- as.factor(y)
  
  x <- as.data.frame(x)
  df <- data.frame(y = y)
  df <- cbind(df, x)
  
  df
  
  
}


sim_test_train <- function(p_cont,
                           p_bin,
                           n_train,
                           n_test,
                           alpha = rnorm(1),
                           beta = rnorm((p_cont+p_bin))){
  
 df_train <-  simple_simulation(p_cont = p_cont,
                                p_bin = p_bin,
                                n = n_train,
                                alpha = alpha,
                                beta = beta) 
 

 
 df_test <- simple_simulation(p_cont = p_cont,
                              p_bin = p_bin,
                              n = n_test,
                              alpha = alpha,
                              beta = beta) 
  
  out <- list(df_train = df_train,
              df_test = df_test)
  
  out
}

n_train <- n_test <- n
p_cont <- p_bin <- 2

sim_list <- sim_test_train(p_cont = p_cont,
                           p_bin = p_bin,
                           n_train = n_train,
                           n_test = n_test,
                           alpha = rnorm(1),
                           beta = rnorm((p_cont+p_bin)))

df_test <- sim_list$df_test
df_train <- sim_list$df_train


covs <- names(df_test)[-1]
outcome_var <- "y"



beta <- rnorm(p)

alpha <- .4

mu <- locfit::expit(alpha + x%*%beta)

y <- rbinom(n, size = 1, prob = mu)
y <- as.factor(y)

x <- as.data.frame(x)

df <- data.frame(y = y)
df <- cbind(df, x)

covs <- names(x)

mod_prep_list <- model_prep(df = df_train,
                            training_prop = 0.5,
                            sampling_group_var,
                            n_folds = 4)



