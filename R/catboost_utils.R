catboost_optimize_parallel_cpu <- function(df, target_var, param_grid, n_cv_sets, train_rate, cat_features)
{

  require(AUC)
  
  df <- data.frame(df)
  
  print("Note: function uses all variables")
  
  param_grid$auc_mean <- rep(0, nrow(param_grid))
  param_grid$auc_var <- rep(0, nrow(param_grid))
  
  target_var_position = which(colnames(df)==target_var)
  
  for (param_set_index in 1:nrow(grid))
  {
    param_set = as.list(param_grid[param_set_index,])
    
    params = append(list(task_type="CPU"), param_set)
    
    params = list(iterations=10, depth=10, loss_function='Logloss')
    
    print(paste("Optimizing param set ", param_set_index, " (", paste(as.character(params),collapse=" "), ")", sep=""))
    
    cv_results = data.frame(round=1:n_cv_sets, auc=rep(0,n_cv_sets))
    
    for (cv_set in 1:cv_sets)
    {
      
      print(paste("CV fold ", cv_set,"/",cv_sets," for param set ",param_set_index,"/",nrow(param_grid),sep=""))
      
      train_index <- createDataPartition(df[,target_var], p=train_rate, list=F)
      train_df <- df[train_index,]
      test_df <- df[-train_index,]
      
      train_pool = catboost.load_pool(data=train_df[,-target_var_position], label=train_df[,target_var], cat_features = cat_features)
      test_pool = catboost.load_pool(data=test_df[,-target_var_position], label=test_df[,target_var], cat_features = cat_features)
      
      model <- catboost.train(learn_pool=train_pool, test_pool=test_pool, params=params)
      test_df$score <- catboost.predict(model, test_pool, prediction_type="Probability")
      cv_results$auc[cv_set] <- AUC::auc(AUC::roc(test_df$score, factor(test_df$target)))
      
    }
    
    param_grid$auc_mean[param_set_index] <- mean(cv_results$auc)
    param_grid$auc_var[param_set_index] <- var(cv_results$auc)

  }
  
  return (param_grid)
  
}