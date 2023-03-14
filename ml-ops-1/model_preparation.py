import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from functools import reduce
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

import consts
from utils import print_log, save_data

N_FOLDS = 5

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
    
def rmsle_cv(model, model_name:str =None):
    kf = KFold(N_FOLDS, shuffle=True, random_state=consts.RANDOM_STATE).get_n_splits(x_train.values)
    rmse = np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    print(f"{model_name if model_name else model} \n\tRMSE mean: {rmse.mean():.4f} \n\tRMSE std: {rmse.std():.4f}")
    return(rmse)

print_log('Data read')

X: pd.DataFrame = pd.read_pickle(consts.X_TRAIN_FULL)
Y: pd.DataFrame = pd.read_pickle(consts.Y_TRAIN_FULL)

print_log('Models init')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=consts.DATA_TEST_SIZE, random_state=consts.RANDOM_STATE)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=consts.RANDOM_STATE))
enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, 
                                loss='huber', random_state =consts.RANDOM_STATE)

xgb.set_config(verbosity=0)
model_xgb = xgb.XGBRegressor(gamma = 0.0468, 
                            learning_rate = 0.05, max_depth = 3, 
                            min_child_weight=1.7817, n_estimators = 2200,
                            reg_alpha = 0.4640, reg_lambda=0.8571,
                            subsample = 0.5213, silent=1,
                            random_state = consts.RANDOM_STATE, nthread = -1)


model_lgb = lgb.LGBMRegressor(verbose=-1, silent = True, objective = 'regression', num_leaves = 5,
                            learning_rate = 0.05, n_estimators = 720,
                            max_bin = 55, bagging_fraction = 0.8,
                            bagging_freq = 5, feature_fraction = 0.2319,
                            feature_fraction_seed=9, bagging_seed = 9,
                            min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)

models = [lasso, enet, krr, gboost, model_xgb, model_lgb]

print_log('Models testing:')
results = list(zip(map(lambda model: rmsle_cv(model), models), models))
model = reduce(lambda res1, res2: res1 if np.average(res1[0]) < np.average(res2[0]) else res2, results)[1]

print_log('Fit best model')
model.fit(X, Y)

print_log(f"Save model: {model}")
save_data(model, consts.MODEL, consts.MODEL_FULL)
