# Model Utility Functions ----------------

# Housekeeping -------------

`%>%` <- dplyr::`%>%`
`%do%` <- foreach::`%do%`

# Model Formulation function ------------------
# turn outcome and covariates into a model formula


model_formula_function <- function(outcome, covariates){
  
  mod_form <- as.formula(paste(outcome,
                               paste(covariates,collapse = "+"),
                               sep = "~")
                         
  )
  
  
}


# Prep data function ---------------------

# This function creates the train and test data set
# as well as creates some necessary tidymodels files like
# the test metrics

model_prep <- function(df,
                       training_prop,
                       sampling_group_var,
                       n_folds,
                       ws = 1,
                       aug_fac = FALSE,
                       y_num = NULL,
                       y_factor = NULL){
  
  if(length(ws) ==1){
    ws <- rep(1,nrow(df))
  }
  
  if(aug_fac == TRUE){
    df$y_num <- y_num
    df$y_factor <- y_factor
  }
  #df$cs_weights <- hardhat::importance_weights(ws)
  
  # Split the data
  df_split <- rsample::initial_split(data = df,
                                     group = sampling_group_var,
                                     prop = training_prop)
  
  # Extract the test and train
  train <- rsample::training(df_split)
  test <- rsample::testing(df_split)
  
  # Create folds within the training data
  train_nfold <- train %>%
    rsample::vfold_cv(n_folds)
  
  # Set the test metrics
  
  mset_bin <- yardstick::metric_set( yardstick::roc_auc,
                                     yardstick::mn_log_loss) 
  
  
  control <- tune::control_grid(save_workflow = TRUE,
                                save_pred = TRUE,
                                extract = tune::extract_model) 
  
  out = list(
    df_split = df_split,
    train = train,
    test = test,
    train_nfold = train_nfold,
    mset_bin = mset_bin,
    control = control
  )
  
  out
}

# Recipe Function -------------------------


model_recipe_function <- function(model_formula,
                                  train,
                                  impute = FALSE,
                                  cov_impute_median,
                                  cov_impute_knn,
                                  knn_impute_with,
                                  n_neighbours){
  if(impute ==TRUE){
    out <- recipes::recipe(model_formula, data = train) %>%
      recipes::step_impute_median(dplyr::any_of(cov_impute_median)) %>%
      recipes::step_impute_knn(dplyr::any_of(cov_impute_knn),
                               impute_with = knn_impute_with,
                               neighbors = n_neighbours)%>%
      recipes::step_dummy(recipes::all_nominal_predictors()) %>% 
      recipes:: step_zv(recipes::all_predictors()) %>% 
      recipes::step_normalize(recipes::all_predictors())
  }else{
    
    out  <- recipes::recipe(model_formula, data = train) %>%
      recipes:: step_dummy(recipes::all_nominal_predictors()) %>% 
      recipes::step_zv(recipes::all_predictors()) %>% 
      recipes::step_normalize(recipes::all_predictors())
    
  }
  
  out
}


#train %>% dplyr::select(cs_weights)
