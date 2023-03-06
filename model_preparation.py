from functools import reduce
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

import consts

N_FOLDS = 5

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
    
def rmsle_cv(model, model_name:str =None):
    kf = KFold(N_FOLDS, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse = np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    print(f"{model_name if model_name else model} score: {rmse.mean():.4f} ({rmse.std():.4f})\n")
    return(rmse)

X: pd.DataFrame = pd.read_pickle(consts.X_TRAIN_FULL)
Y: pd.DataFrame = pd.read_pickle(consts.Y_TRAIN_FULL)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=consts.DATA_TEST_SIZE, random_state=consts.RANDOM_STATE)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, 
                                loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                            learning_rate=0.05, max_depth=3, 
                            min_child_weight=1.7817, n_estimators=2200,
                            reg_alpha=0.4640, reg_lambda=0.8571,
                            subsample=0.5213, silent=1,
                            random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                            learning_rate=0.05, n_estimators=720,
                            max_bin = 55, bagging_fraction = 0.8,
                            bagging_freq = 5, feature_fraction = 0.2319,
                            feature_fraction_seed=9, bagging_seed=9,
                            min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                meta_model = lasso)

models = [lasso, ENet, KRR, GBoost, model_xgb, model_lgb, averaged_models, stacked_averaged_models]
results = list(zip(map(lambda model: rmsle_cv(model), models), models))
model = reduce(lambda res1, res2: res1 if np.average(res1[0]) > np.average(res2[0]) else res2, results)[1]
model.fit(X, Y)

Path(consts.MODEL).mkdir(parents=True, exist_ok=True)
with open(consts.MODEL_FULL, 'wb') as f:
  pickle.dump(model, f)
