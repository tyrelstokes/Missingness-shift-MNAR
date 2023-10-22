# Fit the weights with xg boost

source(here::here("vae/outcome-models/xg-boost-model.R"))
source(here::here("vae/outcome-models/model-utility-functions.R"))


weight_function <- function(df_source,
                            df_target,
                            R_source,
                            R_target,
                            trees_vec = c(4,6,8,10,12),
                            mtry_vec = c(1,2,4,8, floor(sqrt(120))),
                            tree_depth_vec = c(2,4,6,8,10),
                            learn_rate_vec = c(.01,.05,.1,.2),
                            loss_reduction_vec = 0,
                            min_n_vec = c(1,2,5,10),
                            outcome_var = "source_factor",
                            covs,
                            model_label,
                            tune_metric = "mn_log_loss",
                            training_prop,
                            n_folds){
  
  
  R_total <- rbind(R_source,R_target)
  
  names(R_total) <- paste0("R_",names(R_total))
  
  n_tot <- nrow(R_total)
  ns <- apply(R_total,2,function(x){sum(x == 1)})
  
 rm_cols <- which(ns == n_tot)
 
 R_total <- R_total[,-rm_cols]
  
  rcovs <- names(R_total)
  
  df_source$source <- 1
  df_target$source <- 0
  

  
  df <- rbind(df_source,df_target)
  
  df <- cbind(df,R_total)
  
  df$source_factor <- as.factor(df$source)
 
  df$y_num <- df$source
  
  mod_prep_list <- model_prep(df = df,
                              training_prop = training_prop,
                              sampling_group_var,
                              n_folds = n_folds,
                              aug_fac = TRUE,
                              y_num = df$source,
                              y_factor = df$source_factor)
  
  
  total_covs <- c(covs,rcovs)
  

 out_list <-  xg_boost_classification(mod_prep_list = mod_prep_list,
                                      trees_vec = trees_vec,
                                      mtry_vec = mtry_vec,
                                      tree_depth_vec = tree_depth_vec,
                                      learn_rate_vec = learn_rate_vec,
                                      loss_reduction_vec = loss_reduction_vec,
                                      min_n_vec = min_n_vec,
                                      outcome_var = outcome_var,
                                      covs = total_covs,
                                      model_label = model_label,
                                      tune_metric = tune_metric)
 
 fit_final <- out_list$fit_final
 
 predictions <- fit_final %>%
   broom::augment(df, type.predict = "response")
 
 preds2 <- fit_final %>%
   broom::augment(df,type.predict = "link")
 
 
 preds2$p_source <- as.numeric(preds2$.pred_1)
 preds2$p_target <- as.numeric(preds2$.pred_0)
 preds2$imp_weight <- preds2$p_target/preds2$p_source
   
 
# predictions$p_source <- as.numeric(predictions$y)
 #predictions$p_target <- 1 - predictions$p_source
 #predictions$imp_weight <- predictions$p_target/predictions$p_source
 
 
 out_list[["predictions"]] <- preds2
#out_list <- append(predictions,out_list)
 
 out_list
  
  
}


