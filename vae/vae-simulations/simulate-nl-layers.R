# housecleaning ----------------

`%do%` <- foreach::`%do%`

# intialize some data ----------------
#z <- data.frame(z1 = full_data$z_1,
              #  z2 = full_data$z_2,
              #  z3 = full_data$z_3)

# Helper functions ----------------


matrix_maker <- function(node_list,
                         ind,
                         n_nodes,
                         n){
  
  out <- matrix(nrow = n, ncol = n_nodes)
  
  
  for(i in 1:n_nodes){
    out[,i] <- node_list[[i]][[ind]]
  }
  out
  
}


list_of_matrix <- function(node_list,
                           n_nodes,
                           n,
                           n_vars){
  
  out_list <- vector('list',length = n_vars)
  
  for(j in 1:n_vars){
    
    out_list[[j]] <- matrix_maker(node_list = node_list,
                                  ind = j,
                                  n_nodes = n_nodes,
                                  n = n)
    
  }
  
  out_list
}

# Sim one node -------------

sim_single_node <- function(parent_var,
                             C_mat,
                             Beta_mat,
                             n_vars,
                             act_fun = "sigmoid"){
  
  
  lin_pred_list <- lin_pred_fun(n_vars = n_vars,
                           parent_var = parent_var,
                           C_mat = C_mat,
                           Beta_mat = Beta_mat)
  
  
  
  if(act_fun == "sigmoid"){
    
   out <-  lapply(lin_pred_list,function(x){
     ns <- ncol(x)
     n1 <- nrow(x)
     out_in <- rep(0,n1)
     for(k in 1:ns){
       out_in <- out_in + x[,k]
     }
     
     if(sum(out_in == 0)< length(n1)){
     
     locfit::expit(out_in)
     }else{
       out_in
     }
    
  })
  
  }
  if(act_fun =="relu"){
    
    out <-  lapply(lin_pred_list,function(x){
      x <- as.matrix(x)
      ns <- ncol(x)
      n1 <- nrow(x)
      out_in <- rep(0,n1)
      for(k in 1:ns){
        out_in <- out_in + x[,k]
      }
      
      if(sum(out_in == 0)< length(n1)){
        
        sigmoid::relu(out_in)
      }else{
        out_in
      }
      
    })
  
  }
out  
}

# layer value function --------------

layer_value <- function(node_mat_list,
                        n_vars,
                        n_nodes,
                        Beta_mat,
                        act_fun = "linear"){
  
  val_list <- vector('list', length = n_vars)
  for(i in 1:n_vars){
    node_mat <- node_mat_list[[i]]
    b_vec <- Beta_mat[i,]
    val <- node_mat %*%b_vec
    alpha <- rnorm(1,mean =0,sd =1)
    if(act_fun =="linear"){
      val_list[[i]] <- val + alpha
    }
    if(act_fun =="sigmoid"){
      val_list[[i]] <- locfit::expit(val + alpha)
    }
  }
  
  val_list
}

# Sim several nodes ------------
sim_single_layer <- function(parent_var,
                            C_mat,
                            n_vars,
                            act_fun = "relu",
                            n_nodes,
                            n,
                            output_act_fun = "linear",
                            rescale = TRUE
                            ){
  
  
  n_p <- ncol(parent_var)
  
 node_list <- vector('list', length = n_nodes)
 beta_list <- vector('list', length = n_nodes)

  
 foreach::foreach(i = 1:n_nodes) %do%{
   
   mn_beta <- rnorm(n_p*n_vars)
   sd_beta <- rgamma(n_p*n_vars,rate = 2,shape =3)
   Beta_mat <- matrix(rnorm(n_p*n_vars, mean = mn_beta,sd = sd_beta),
                      nrow = n_p,
                      ncol = n_vars)
   
   node_list[[i]] <-  sim_single_node(parent_var = parent_var,
                            C_mat = C_mat,
                            Beta_mat = Beta_mat,
                            n_vars = n_vars,
                            act_fun = act_fun)
   
  beta_list[[i]] <- Beta_mat
   
   
 }

 node_mat_list <- list_of_matrix(node_list = node_list,
                                 n_nodes = n_nodes,
                                 n = n,
                                 n_vars = n_vars)

  
 node_beta_mat <-  matrix(rnorm(n_nodes*n_vars),
                          nrow = n_vars,
                          ncol = n_nodes)
 
 
 layer_val <- layer_value(node_mat_list = node_mat_list,
                            n_vars = n_vars,
                            n_nodes = n_nodes,
                            Beta_mat = node_beta_mat,
                            act_fun = output_act_fun)
 
 if(rescale == TRUE){
   layer_val <- lapply(layer_val,function(x){
     if(sum(x == mean(x))<length(x)){
     x <- (x - mean(x))/sd(x)
     }
     x-mean(x)
   })
 }
  
 layer_val
}





# simulate several layers -----------------------  

sim_many_layers <-   function(parent_var,
                              C_mat,
                              n_vars,
                              act_fun = "sigmoid",
                              output_act_fun = "linear",
                              n_nodes,
                              n_layers,
                              final_output_act_fun = "linear",
                              n,
                              rescale = TRUE){
  
  
  n_p <- ncol(parent_var)
  
  layer_list <- vector('list', length = n_layers)

  
  foreach::foreach(i = 1:n_layers) %do%{
    
  
    
    layer_list[[i]] <-  sim_single_layer(parent_var = parent_var,
                         C_mat = C_mat,
                         n_vars = n_vars,
                         act_fun = act_fun,
                         n_nodes = n_nodes[i],
                         n = n,
                         output_act_fun = output_act_fun,
                         rescale = rescale
    )
  
    
  }
  
  layer_mat_list <- list_of_matrix(node_list = layer_list,
                                   n_nodes = n_layers,
                                   n = n,
                                   n_vars = n_vars)
  
  
  layer_beta_mat <-  matrix(rnorm(n_layers*n_vars, sd = 2.5),
                            nrow = n_vars,
                            ncol = n_layers)
  
  
  layer_val <- layer_value(node_mat_list = layer_mat_list,
                           n_vars = n_vars,
                           n_nodes = n_nodes,
                           Beta_mat = layer_beta_mat,
                           act_fun = output_act_fun)
  
  if(rescale ==TRUE){
    layer_val <- lapply(layer_val,function(x){
      if(sum(x==mean(x))<length(x)){
      x <- (x - mean(x))/sd(x)
      }
      x - mean(x)
    })
    
  }
  
  #xb <- Reduce(f = "+",layer_val)
  var_mat <- do.call(cbind,layer_val)
  
  out <- list(value_list = layer_val,
              var_mat = var_mat)
  
out  
}





#sim_many_layers(parent_var = as.matrix(z[,1]),
 #               C_mat = as.matrix(C_mat_x_z[,2]),
  #              n_vars = 1,
   #             act_fun = "relu",
    #            output_act_fun = "linear",
     #           n_nodes = c(3,3,3),
      #          n_layers = 3,
       #         final_output_act_fun = "linear",
        #        n = n,
         #       rescale = TRUE)

