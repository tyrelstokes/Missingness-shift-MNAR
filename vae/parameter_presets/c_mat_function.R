
assign_function <- function(object,
                            source = FALSE,
                            target = FALSE){
  
 nm <- deparse(substitute(object))
 
 assign(x = nm,
        value = object,
        envir = .GlobalEnv)
 
 if(source == TRUE){
   nm_source <- paste0(nm,"_source")
   assign(x = nm_source,
          value = object,
          envir = .GlobalEnv)
 }
 
 if(target == TRUE){
   nm_target <- paste0(nm,"_target")
   
   assign(x = nm_target,
          value = object,
          envir = .GlobalEnv) 
   
   
 }
 
 
 
}






C_mat_function <- function(z_causes_x,
                           x_causes_x,
                           x_causes_r,
                           r_causes_r,
                           x_causes_y,
                           n_z,
                           n_x,
                           n_r,
                           source = FALSE,
                           target = FALSE){
  
 C_mat_x_z <- matrix(rep(z_causes_x,n_z),nrow = n_z)
 C_mat_x_x <- matrix(rep(x_causes_x,n_x),nrow = n_x)
 diag(C_mat_x_x) <- rep(0,n_x)
 C_mat_x_x[lower.tri(C_mat_x_x)] <- rep(0,3)
 
 C_mat_r_x <- matrix(rep(x_causes_r,n_x),nrow = n_x)
 
 C_mat_r_r <- matrix(rep(r_causes_r,n_r),nrow = n_r)
 diag(C_mat_r_r) <- rep(0,n_r)
 C_mat_r_r[lower.tri(C_mat_r_r)] <- rep(0,3)
 
 C_mat_y_x <- matrix(x_causes_y,nrow = n_x)
 
 assign_function(C_mat_x_z, source = source, target = target)
 assign_function(C_mat_x_x, source = source, target = target)
 assign_function(C_mat_r_x, source = source, target = target)
 assign_function(C_mat_r_r, source = source, target = target)
 assign_function(C_mat_y_x, source = source, target = target)
 

 
 
 
 
}
  
  
  
