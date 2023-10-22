# Housekeeping ------------------
`%>%` <- dplyr::`%>%`

# RMSE plot --------------------
rmse_plot <- function(rmse_frame,
                      ipsum = FALSE){
  
 p1 <- ggplot2::ggplot(rmse_frame,ggplot2::aes(x = factor(model_name, level = c("Z model",
                                                                                "Joint Model",
                                                                                "No outcome model", 
                                                                                "No missingness model")),
                                          y = rmse,
                                          fill = model_name)) +
    ggplot2::geom_boxplot()+
    viridis::scale_fill_viridis(discrete = TRUE, alpha=0.6) +
    ggplot2::geom_jitter(color="grey", size=0.3, alpha=0.1) +
    ggplot2::theme(
      legend.position="none",
      plot.title = ggplot2::element_text(size=11)
    ) +
    ggplot2::ggtitle("RMSE by Model") +
    ggplot2::xlab("")+
    ggplot2::geom_hline(yintercept = 1)
 
 if(ipsum ==TRUE){
   p1 + hrbrthemes::theme_ipsum()
 }else{
   p1
 }
  
}

# bias plot ---------------------------

bias_plot <- function(rmse_frame,
                      ipsum = FALSE){
  
  p2 <- ggplot2::ggplot(rmse_frame,ggplot2::aes(x = factor(model_name, level = c("Z model",
                                                                                 "Joint Model",
                                                                                 "No outcome model",
                                                                                 "No missingness model")),
                                          y = bias,
                                          fill = model_name)) +
    ggplot2::geom_boxplot()+
    viridis::scale_fill_viridis(discrete = TRUE, alpha=0.6) +
    ggplot2::geom_jitter(color="grey", size=0.3, alpha=0.1) +
    ggplot2::theme(
      legend.position="none",
      plot.title = ggplot2::element_text(size=11)
    ) +
    ggplot2::ggtitle("Bias by Model") +
    ggplot2::xlab("")+
    ggplot2::geom_hline(yintercept = 0)
  
  if(ipsum ==TRUE){
    p2 + hrbrthemes::theme_ipsum()
  }else{
    p2
  }
}

# Summary function -------------------------------

summary_fun <- function(rmse_frame){
  
 rmse_frame <- as.data.frame(rmse_frame)
 rmse_frame %>%
   dplyr::filter(type =="combined") %>%
   dplyr::group_by(model_name) %>%
   dplyr::summarise(mn_rmse = mean(rmse),
                    mn_bias = mean(bias),
                     sample = dplyr::n())
  
  
}


# RMSE plot --------------------
rmse_plot_da <- function(rmse_frame,
                         type,
                         split_by_data = FALSE,
                         ipsum = FALSE){
  
  p1 <- ggplot2::ggplot(rmse_frame,ggplot2::aes(x = factor(model_name, level = c("Z model",
                                                                                 "Joint Model",
                                                                                 "No outcome model", 
                                                                                 "No missingness model")),
                                                y = rmse,
                                                fill = model_name)) +
    ggplot2::geom_boxplot()+
    viridis::scale_fill_viridis(discrete = TRUE, alpha=0.6) +
    ggplot2::geom_jitter(color="grey", size=0.3, alpha=0.1) +
    ggplot2::theme(
      legend.position="none",
      plot.title = ggplot2::element_text(size=11)
    ) +
    ggplot2::ggtitle(paste(type,"RMSE by Model")) +
    ggplot2::xlab("")+
    ggplot2::geom_hline(yintercept = 1)
  
  if(split_by_data == TRUE){
    p1 <- p1 + ggplot2::facet_wrap(~data_source)
  }
  
  if(ipsum ==TRUE){
    p1 + hrbrthemes::theme_ipsum()
  }else{
    p1
  }
  
}


# RMSE plot --------------------
rmse_plot_gen <- function(rmse_frame,
                          type,
                         split_by_data = FALSE,
                         ipsum = FALSE,
                         mod_order,
                         filter_out = FALSE){
  
  
  rmse_frame <- as.data.frame(rmse_frame)
  if(filter_out ==TRUE){
    rmse_frame <- rmse_frame %>% 
      dplyr::filter(type == "combined")
  }
  
  p1 <- ggplot2::ggplot(rmse_frame,ggplot2::aes(x = factor(model_name, level = mod_order),
                                                y = rmse,
                                                fill = model_name)) +
    ggplot2::geom_boxplot()+
    viridis::scale_fill_viridis(discrete = TRUE, alpha=0.6) +
    ggplot2::geom_jitter(color="grey", size=0.3, alpha=0.1) +
    ggplot2::theme(
      legend.position="none",
      plot.title = ggplot2::element_text(size=11)
    ) +
    ggplot2::ggtitle(paste(type,"RMSE by Model")) +
    ggplot2::xlab("")+
    ggplot2::geom_hline(yintercept = 1)
  
  if(split_by_data == TRUE){
    p1 <- p1 + ggplot2::facet_wrap(~data_type)
  }
  
  if(ipsum ==TRUE){
    p1 + hrbrthemes::theme_ipsum()
  }else{
    p1
  }
  

}

# Bias plot  gen --------------------
bias_plot_gen <- function(rmse_frame,
                          type,
                          split_by_data = FALSE,
                          ipsum = FALSE,
                          mod_order,
                          filter_out = FALSE){
  
  if(filter_out ==TRUE){
    rmse_frame <- rmse_frame %>% 
      dplyr::filter(type == "combined")
  }
  
  p1 <- ggplot2::ggplot(rmse_frame,ggplot2::aes(x = factor(model_name, level = mod_order),
                                                y = bias,
                                                fill = model_name)) +
    ggplot2::geom_boxplot()+
    viridis::scale_fill_viridis(discrete = TRUE, alpha=0.6) +
    ggplot2::geom_jitter(color="grey", size=0.3, alpha=0.1) +
    ggplot2::theme(
      legend.position="none",
      plot.title = ggplot2::element_text(size=11)
    ) +
    ggplot2::ggtitle(paste(type,"Bias by Model")) +
    ggplot2::xlab("")+
    ggplot2::geom_hline(yintercept = 1)
  
  if(split_by_data == TRUE){
    p1 <- p1 + ggplot2::facet_wrap(~data_source)
  }
  
  if(ipsum ==TRUE){
    p1 + hrbrthemes::theme_ipsum()
  }else{
    p1
  }
  
}

# save plot fun ------------------

save_plot <- function(gplot,
                      fname){
  ind <- TRUE
  indexer <- 0
  while(ind ==TRUE){
    indexer <- indexer + 1
    full_name <- paste0(fname,paste0("_",
                                     paste0(indexer,".png")))
    total_name <- here::here(paste0("vae/server-files/plots/",
                                    full_name))
    ind <- file.exists(total_name)
    
  }
  
  ggplot2::ggsave(total_name,plot = gplot)
}

# save data fun ---------

save_data <- function(dt,
                      fname,
                      findit = TRUE,
                      num = 0,
                      ext = ".csv"){
  
  if(findit == TRUE){
  ind <- TRUE
  indexer <- 0
  while(ind ==TRUE){
    indexer <- indexer + 1
    full_name <- paste0(fname,paste0("_",
                                     paste0(indexer,ext)))
    
    total_name <- here::here(paste0("vae/server-files/data-frames/",
                                    full_name))
    ind <- file.exists(total_name)
   
  }
  

  
  }else{
    
    
    full_name <- paste0(fname,paste0("_",
                                     paste0(num,ext)))
    
    total_name <- here::here(paste0("vae/server-files/data-frames/",
                                    full_name))
    
    indexer <- 0
    
  }
    
  
  if(ext == ".csv"){
  write.csv(dt,file = total_name)
  }
  
  if(ext ==".Rda"){
    saveRDS(dt,file = total_name)
  }
  
 
  return(indexer)
  
}

# merge meta_df and rmse

merge_meta_rmse <- function(meta_df,
                            rmse_frame){
  
  rmse_frame$N <- meta_df$N
  rmse_frame$n <- meta_df$n
  rmse_frame$target_prop <- meta_df$target_prop
  rmse_frame$miss_prob_2 <- meta_df$miss_prob_2
  rmse_frame$miss_prob_3 <- meta_df$miss_prob_3
  rmse_frame$dag <- meta_df$dag
  rmse_frame$x_link_fun <- meta_df$x_link_fun
  rmse_frame$y_link_fun <- meta_df$y_link_fun
  rmse_frame$beta_sd <- meta_df$beta_sd
  
  rmse_frame$act_fun_y <- meta_df$act_fun_y
  rmse_frame$final_act_y <- meta_df$final_act_y
  rmse_frame$act_fun_x <- meta_df$act_fun_x
  rmse_frame$final_act_x <- meta_df$final_act_x
  
  rmse_frame$n_nodes_x <- meta_df$n_nodes_x
  rmse_frame$n_layers_x <- meta_df$n_layers_x
  rmse_frame$n_nodes_y <- meta_df$n_nodes_y
  rmse_frame$n_layers_y <- meta_df$n_layers_y
  
  
  
  
  
  rmse_frame
  
}


