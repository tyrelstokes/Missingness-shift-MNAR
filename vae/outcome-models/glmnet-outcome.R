# Housekeeping

`%>%` <- dplyr::`%>%`
source(here::here("vae/outcome-models/model-utility-functions.R"))

# logloss function ----------------

LogLoss <- function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}


# outcome model ---------------


glmnet_outcome <- function(train,
                           test,
                           outcome_var,
                           covs,
                           ws,
                           n_folds,
                           y_factor = TRUE){



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

#

train  <- train %>%
  dplyr::mutate(ethnicity_aa = ifelse(is.na(ethnicity_aa),0,ethnicity_aa),
                ethnicity_asian = ifelse(is.na(ethnicity_asian),0,ethnicity_asian),
                ethnicity_hispanic = ifelse(is.na(ethnicity_hispanic),0,ethnicity_hispanic),
                ethnicity_na = ifelse(is.na(ethnicity_na),0,ethnicity_na),
                ethnicity_other_unknown = ifelse(is.na(ethnicity_other_unknown),0,ethnicity_other_unknown))

test  <- test %>%
  dplyr::mutate(ethnicity_aa = ifelse(is.na(ethnicity_aa),0,ethnicity_aa),
                ethnicity_asian = ifelse(is.na(ethnicity_asian),0,ethnicity_asian),
                ethnicity_hispanic = ifelse(is.na(ethnicity_hispanic),0,ethnicity_hispanic),
                ethnicity_na = ifelse(is.na(ethnicity_na),0,ethnicity_na),
                ethnicity_other_unknown = ifelse(is.na(ethnicity_other_unknown),0,ethnicity_other_unknown))

#create the sparse matrices
train_sparse <- Matrix::sparse.model.matrix(model_formula, data = train)[,-1] 
test_sparse <- Matrix::sparse.model.matrix(model_formula, data = test)[,-1] 
# valid_sparse <- Matrix::sparse.model.matrix(model_formula,data = test_final)[,-1]


options(na.action=previous_na_action$na.action) #reset the na.action


y_out <- train %>% dplyr::select(dplyr::one_of(outcome_var)) %>%
  as.data.frame()
if(y_factor ==TRUE){
  y_out_num <- as.numeric(y_out[[1]])
  y_out_factor <- ifelse(y_out_num ==1, 0,1)
}


glmcv_weight <- glmnet::cv.glmnet(x = train_sparse,y = y_out_factor,
                           weights = ws,
                           nfolds = n_folds,
                           family = "binomial",
                           type.measure = "deviance")


glmcv <- glmnet::cv.glmnet(x = train_sparse,y = y_out_factor,
                                  nfolds = n_folds,
                                  family = "binomial",
                                  type.measure = "deviance")


lmin <- glmcv$lambda.min
l1se <- glmcv$lambda.1se


lmin_w <- glmcv_weight$lambda.min
l1se_w <- glmcv_weight$lambda.1se


preds <- predict(glmcv,
                 newx = test_sparse,
                 s = "lambda.min",
                 type = "response")


preds_w <- predict(glmcv_weight,
                   newx = test_sparse,
                   s = "lambda.min",
                   type = "response")


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



test_rmse_w <- sqrt(mean((y_out_test_factor - preds_w)^2))
# valid_rmse <- sqrt(mean((y_out_test_num - preds)^2))

lloss_w <- LogLoss(actual = y_out_test_factor,
                 predicted = preds_w)


loss_df <- data.frame(metric = c(rep("rmse",2),
                                 rep("log-loss",2)),
                      val = c(test_rmse,test_rmse_w,
                              lloss,lloss_w),
                      weighted = c(F,T,F,T))


out <- list(loss_df = loss_df,
            preds = preds,
            preds_w = preds_w,
            glmcv = glmcv,
            glmcv_weight = glmcv_weight)


out

}






