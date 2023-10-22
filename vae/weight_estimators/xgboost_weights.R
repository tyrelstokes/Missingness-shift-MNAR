# Source the model utility functions-----------

`%>%` <- dplyr::`%>%`

source(here::here("vae/outcome-models/model-utility-functions.R"))

# logloss function ----------------

LogLoss <- function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}


# make the weighted xg boost model function ------

weight_xg_boost_classification <- function(train,
                                           test,
                                    trees_vec = c(4,6,8,10,12),
                                    mtry_vec = c(1,2,4,8, floor(sqrt(length(120)))),
                                    tree_depth_vec = c(2,4,6,8,10),
                                    learn_rate_vec = c(.01,.05,.1,.2,.3),
                                    loss_reduction_vec = 0,
                                    col_sample_vec = c(.1,.5,1),
                                    lambda_vec = 1,
                                    alpha_vec = 0,
                                    outcome_var,
                                    covs,
                                    model_label,
                                    tune_metric = "mn_log_loss",
                                    ws = 1,
                                    n_folds = 5,
                                    nrounds = 300,
                                    y_factor = FALSE,
                                    strat = TRUE){
  
  
  
  # Create the model Formula
  
  model_formula <- model_formula_function(outcome = outcome_var,
                                          covariates = covs)
  
  # Extract the prepped data 
 # df_split <- mod_prep_list$df_split
 #  train <- mod_prep_list$train
#  test <- mod_prep_list$test
#  train_nfold <- mod_prep_list$train_nfold
  
  #train$cs_weights <- hardhat::importance_weights(train$cs_weights)
  
  
  previous_na_action <- options('na.action') #store the current na.action
  options(na.action='na.pass') #change the na.action
  
  #create the sparse matrices
  train_sparse <- Matrix::sparse.model.matrix(model_formula, data = train)[,-1] 
  test_sparse <- Matrix::sparse.model.matrix(model_formula, data = test)[,-1] 
 # valid_sparse <- Matrix::sparse.model.matrix(model_formula,data = test_final)[,-1]
  
  options(na.action=previous_na_action$na.action) #reset the na.action
  
  
  # Set the hyper-parameters
  
  # booster = 'gbtree': Possible to also have linear boosters as your weak learners.
  params_booster <- list(booster = 'gbtree',
                         eta = learn_rate_vec,
                         gamma = loss_reduction_vec,
                         max.depth = tree_depth_vec,
                         subsample = 1,
                         colsample_bytree = col_sample_vec,
                         min_child_weight = mtry_vec,
                         alpha = alpha_vec,
                         lambda = lambda_vec,
                         objective = "binary:logistic")
  
  y_out <- train %>% dplyr::select(dplyr::one_of(outcome_var)) %>%
    as.data.frame()
  if(y_factor ==TRUE){
  y_out_num <- as.numeric(y_out[[1]])
  y_out_factor <- ifelse(y_out_num ==1, 0,1)
  }
   
  bst.cv <- xgboost::xgb.cv(data = train_sparse, 
                            label = y_out_factor, 
                            params = params_booster,
                            nrounds = nrounds, 
                            nfold = n_folds,
                            print_every_n = 20,
                            verbose = 2,
                            weight = ws,
                            early_stopping_rounds = 50,
                            stratified = TRUE)
  
  
  
  res_df <- data.frame(TRAINING_ERROR = bst.cv$evaluation_log$train_logloss_mean, 
                       VALIDATION_ERROR = bst.cv$evaluation_log$test_logloss_mean, # Don't confuse this with the test data set. 
                       ITERATION = bst.cv$evaluation_log$iter) %>%
    dplyr::mutate(MIN = VALIDATION_ERROR == min(VALIDATION_ERROR))
  best_nrounds <- res_df %>%
    dplyr::filter(MIN) %>%
    dplyr::pull(ITERATION)
  res_df_longer <- tidyr::pivot_longer(data = res_df, 
                                cols = c(TRAINING_ERROR, VALIDATION_ERROR), 
                                names_to = "ERROR_TYPE",
                                values_to = "ERROR")
  
  
  
  bstSparse <- xgboost::xgboost(data = train_sparse,
                       label = y_out_factor,
                       nrounds = best_nrounds,
                       params = params_booster,
                       print_every_n = 25,
                       weight = ws)
  
  
  preds <- predict(bstSparse, test_sparse)
  
  #preds_valid <- predict(bstSparse, valid_sparse)
  
  y_out_test <- test %>% dplyr::select(dplyr::one_of(outcome_var)) %>%
   as.data.frame()
  
 # y_out_valid <- test_final %>% dplyr::select(dplyr::one_of(outcome_var)) %>%
  #  as.data.frame()
  
  if(y_factor ==TRUE){
  y_out_test_num <- as.numeric(y_out_test[[1]])
  y_out_test_factor <- ifelse(y_out_test_num ==1, 0,1)
  
 # y_out_valid_num <- as.numeric(y_out_valid[[1]])
#  y_out_valid_factor <- ifelse(y_out_valid_factor ==1, 0,1)
  
  }
  

  test_rmse <- sqrt(mean((y_out_test_factor - preds)^2))
 # valid_rmse <- sqrt(mean((y_out_test_num - preds)^2))
  
  lloss <- LogLoss(actual = y_out_test_factor,
                   predicted = preds)
  
  #valid_lloss <- LogLoss(actual = y_out_valid_num,
                       #  predicted = preds_valid)
  
  
 out <- list(preds = preds,
             #preds_valid = preds_valid,
             y_out_test_num = y_out_test_num,
             y_out_test_factor = y_out_test_factor,
             #y_out_valid_num = y_out_valid_num,
            # y_out_valid_factor = y_out_valid_factor,
             bstSparse = bstSparse,
             bst.cv = bst.cv,
             test = test,
             train = train,
             test_rmse = test_rmse,
             lloss = lloss)
             #valid_rmse = valid_rmse,
             #valid_lloss)
 
 out
  
  
}
