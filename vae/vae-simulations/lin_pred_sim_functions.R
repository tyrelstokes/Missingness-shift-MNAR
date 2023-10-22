# Housekeeping ---------

`%>%` <- dplyr::`%>%`
`%do%` <- foreach::`%do%`

# single lin pred function ------------

lin_pred_once <- function(parent_var,
                          C_vec,
                          Beta_vec){
  
  w_cols <- (C_vec ==1)
  
  if((length(w_cols)>0) &(sum(w_cols) >0)){
    if(is.vector(parent_var) ==TRUE){
      w_eff <- parent_var
    }else{
    w_eff <- parent_var[,w_cols]
    }
    b_eff <- Beta_vec[w_cols]
    
    if(length(b_eff) >1){
     lin_pred <- w_eff %*% b_eff
    }else{
    lin_pred <-  w_eff*b_eff
      
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
    if(sum(C_mat[j,])>0){
    lin_pred <-  lin_pred_once(parent_var = parent_var,
                               C_vec = C_mat[j,],
                               Beta_vec = Beta_mat[j,])
    
    }else{
      lin_pred <- rep(0,nrow(parent_var))
    }
    
    
    lin_pred
  })
  
  lin_pred_list
  
}
