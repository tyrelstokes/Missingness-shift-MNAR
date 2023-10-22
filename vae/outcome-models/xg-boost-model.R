# Outcome Models

source(here::here("vae/outcome-models/model-utility-functions.R"))

# xGBoost Functions ---------------

xg_boost_mod <- function(mod_prep_list,
                         trees_vec,
                         mtry_vec,
                         tree_depth_vec,
                         learn_rate_vec,
                         loss_reduction_vec,
                         min_n_vec,
                         outcome_var,
                         covs,
                         model_label,
                         tune_metric = "mn_log_loss"){
  
  # Create the model Formula
  
  model_formula <- model_formula_function(outcome = outcome_var,
                                          covariates = covs)
  
  # Extract the prepped data 
  df_split <- mod_prep_list$df_split
  train <- mod_prep_list$train
  test <- mod_prep_list$test
  train_nfold <- mod_prep_list$train_nfold
  
  # Extract the metrics and control for tidymodels
  mset_bin <- mod_prep_list$mset_bin
  control <- mod_prep_list$control
  
  
  # Set up the model recipe using the model recipe utility function
  source(here::here("model-files/model-utility-functions.R"))
  
  xg_rec <-  model_recipe_function(model_formula = model_formula,
                                   train = train,
                                   impute = FALSE,
                                   cov_impute_median = NULL,
                                   cov_impute_knn = NULL,
                                   knn_impute_with = NULL,
                                   n_neighbours = NULL)
  
  
  # Set up the xg boost model and which parameters to tune
  xg_model <- parsnip::boost_tree(mode = "regression", #
                                  trees = tune::tune(),
                                  mtry = tune::tune(),
                                  tree_depth = tune::tune(),
                                  learn_rate = tune::tune(),
                                  loss_reduction = tune::tune(),
                                  min_n = tune::tune())
  
  # Set up tidymodels workflow
  
  xg_wf <- 
    workflows::workflow() %>% 
    workflows::add_model(xg_model) %>% 
    workflows::add_recipe(xg_rec)
  
  # Tune the model
  
  TuneResults <- xg_wf %>%
    tune::tune_grid(train_nfold,
                    metrics = mset_bin,
                    control = control,
                    grid = tidytable::crossing(trees = trees_vec,
                                               mtry = mtry_vec, 
                                               tree_depth = tree_depth_vec,
                                               learn_rate = learn_rate_vec,
                                               loss_reduction = loss_reduction_vec,
                                               min_n = min_n_vec))
  
  # Get the tuning plot
  tune_plot <- tune::autoplot(TuneResults)
  tune_table <- tune::collect_metrics(TuneResults)
  # Pick the best penalized logistic regression according to 
  # the cross validated folds according to log-loss
  
  best_params <- tune::select_best(TuneResults,
                                   metric = tune_metric)
  
  # Create the model object based on the best model parameters
  
  best_model <- parsnip::boost_tree(mode = "regression",
                                    trees = best_params$trees,
                                    mtry = best_params$mtry,
                                    tree_depth = best_params$tree_depth,
                                    learn_rate = best_params$learn_rate,
                                    loss_reduction = best_params$loss_reduction,
                                    min_n = best_params$min_n)
  
  
  
  # Create the final model
  
  final_mod <- tune::finalize_workflow(
    workflows::workflow() %>% 
      workflows::add_recipe(xg_rec)
    %>% workflows::add_model(best_model),
    best_params
  )
  
  # Fit the final model
  
  fit_final <- final_mod %>%
    parsnip::fit(train)
  
  # Calculate some accuracy measures on the test data
  pdc_pred <- fit_final %>%
    broom::augment(test, type.predict = "response")
  
  rmse <- pdc_pred %>%
    yardstick::rmse(truth = )
 # roc <- pdc_pred %>%
  #  yardstick::roc_auc(truth = pdc_factor,
   #                    estimate =  .pred_1,
    #                   na_rm = TRUE)
  
 # acc <- pdc_pred %>%
  #  yardstick::accuracy(truth = pdc_factor,
   #                     estimate = .pred_class)
  
  
  out <- list(
    pdc_pred = pdc_pred,
    roc = roc,
    acc = acc,
    mod_prep_list = mod_prep_list,
    model_output =  fit_final,
    TuneResults = TuneResults,
    tune_table = tune_table,
    best_params = best_params,
    outcome_var = outcome_var,
    covs = covs,
    model_formula = model_formula,
    model_label = model_label
  ) 
  
  
}


### Classification

source(here::here("vae/outcome-models/model-utility-functions.R"))

# xGBoost Functions ---------------

xg_boost_classification <- function(mod_prep_list,
                         trees_vec = c(4,6,8,10,12),
                         mtry_vec = c(1,2,4,8, floor(sqrt(length(covs)))),
                         tree_depth_vec = c(2,4,6,8,10),
                         learn_rate_vec = c(.01,.05,.1,.2),
                         loss_reduction_vec = 0,
                         min_n_vec = c(1,2,5,10),
                         outcome_var,
                         covs,
                         model_label,
                         tune_metric = "mn_log_loss",
                         ws = 1){
  
  # Create the model Formula
  
  model_formula <- model_formula_function(outcome = outcome_var,
                                          covariates = covs)
  
  # Extract the prepped data 
  df_split <- mod_prep_list$df_split
  train <- mod_prep_list$train
  test <- mod_prep_list$test
  train_nfold <- mod_prep_list$train_nfold
  
 # train$cs_weights <- hardhat::importance_weights(train$cs_weights)
  
  # Extract the metrics and control for tidymodels
  mset_bin <- mod_prep_list$mset_bin
  control <- mod_prep_list$control
  
  # caseweights
  

  
 # final_weights <- hardhat::importance_weights(ws)
  
  # Set up the model recipe using the model recipe utility function
  #source(here::here("model-files/model-utility-functions.R"))
  
  xg_rec <-  model_recipe_function(model_formula = model_formula,
                                   train = train,
                                   impute = FALSE,
                                   cov_impute_median = NULL,
                                   cov_impute_knn = NULL,
                                   knn_impute_with = NULL,
                                   n_neighbours = NULL)
  
  
  # Set up the xg boost model and which parameters to tune
  xg_model <- parsnip::boost_tree(mode = "classification", #
                                  trees = tune::tune(),
                                  mtry = tune::tune(),
                                  tree_depth = tune::tune(),
                                  learn_rate = tune::tune(),
                                  loss_reduction = tune::tune(),
                                  min_n = tune::tune())
  
  # Set up tidymodels workflow
  
  xg_wf <- 
    workflows::workflow() %>% 
    workflows::add_model(xg_model) %>% 
    workflows::add_recipe(xg_rec)
  
  # Tune the model
  
  TuneResults <- xg_wf %>%
    tune::tune_grid(train_nfold,
                    metrics = mset_bin,
                    control = control,
                    grid = tidytable::crossing(trees = trees_vec,
                                               mtry = mtry_vec, 
                                               tree_depth = tree_depth_vec,
                                               learn_rate = learn_rate_vec,
                                               loss_reduction = loss_reduction_vec,
                                               min_n = min_n_vec))
  
  # Get the tuning plot
  tune_plot <- tune::autoplot(TuneResults)
  tune_table <- tune::collect_metrics(TuneResults)
  # Pick the best penalized logistic regression according to 
  # the cross validated folds according to log-loss
  
  best_params <- tune::select_best(TuneResults,
                                   metric = tune_metric)
  
  # Create the model object based on the best model parameters
  
  best_model <- parsnip::boost_tree(mode = "classification",
                                    trees = best_params$trees,
                                    mtry = best_params$mtry,
                                    tree_depth = best_params$tree_depth,
                                    learn_rate = best_params$learn_rate,
                                    loss_reduction = best_params$loss_reduction,
                                    min_n = best_params$min_n)
  
  
  
  # Create the final model
  
  final_mod <- tune::finalize_workflow(
    workflows::workflow() %>% 
      workflows::add_recipe(xg_rec)
    %>% workflows::add_model(best_model),
    best_params
  )
  
  # Fit the final model

  
  fit_final <- final_mod %>%
    parsnip::fit(train)
  
  
  
  # Calculate some accuracy measures on the test data
  y_pred <- fit_final %>%
    broom::augment(test, type.predict = "response")
  
 # y_pred$y_num <- y_num
#  y_pred$y_factor <- y_factor
  
 # y_pred$y_num <- as.numeric(y_pred$y)
  
 rmse <- y_pred %>%
    yardstick::rmse(truth = y_num,
                   estimate = .pred_1)
  
  lloss <- y_pred %>%
    yardstick::rmse(truth = y_num,
                    estimate = .pred_1)
  
   acc <- y_pred %>%
   yardstick::accuracy(truth = y_factor,
                      estimate = .pred_class)
  
   cal_plot <- y_pred %>%
     probably::cal_plot_logistic(truth = y_factor,
                                 estimate = .pred_0,
                                 smooth = FALSE)
  
  out <- list(y_pred = y_pred,
    lloss = lloss,
    acc = acc,
    mod_prep_list = mod_prep_list,
    model_output =  fit_final,
    TuneResults = TuneResults,
    tune_table = tune_table,
    best_params = best_params,
    outcome_var = outcome_var,
    covs = covs,
    model_formula = model_formula,
    model_label = model_label,
    cal_plot = cal_plot,
    final_mod = final_mod,
    fit_final = fit_final
  ) 
  
  out
}







