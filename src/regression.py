import pandas as pd
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

def linear_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    

    # ハイパーパラメータ探索の定義（length）
    def objective_length(trial):
        params = {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'copy_X': trial.suggest_categorical('copy_X', [True, False]),
            'n_jobs': trial.suggest_categorical('n_jobs', [-1, 1, 2]),
            'positive': trial.suggest_categorical('positive', [True, False])
        }

        model = LinearRegression(**params)
        model.fit(X_train, y_length_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, preds, squared=False)
        return rmse

    # ハイパーパラメータ探索の定義（size）
    def objective_size(trial):
        params = {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'copy_X': trial.suggest_categorical('copy_X', [True, False]),
            'n_jobs': trial.suggest_categorical('n_jobs', [-1, 1, 2]),
            'positive': trial.suggest_categorical('positive', [True, False])
        }

        model = LinearRegression(**params)
        model.fit(X_train, y_size_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, preds, squared=False)
        return rmse

    # Optunaによるハイパーパラメータ探索の実行（長さ）
    study_length = optuna.create_study(direction='minimize')
    study_length.optimize(objective_length, n_trials=100)

    # Optunaによるハイパーパラメータ探索の実行（サイズ）
    study_size = optuna.create_study(direction='minimize')
    study_size.optimize(objective_size, n_trials=100)

    # 最適なハイパーパラメータの取得（長さ）
    best_params_length = study_length.best_params

    # 最適なハイパーパラメータの取得（サイズ）
    best_params_size = study_size.best_params

    # 最適なモデルの学習（長さ）
    best_model_length = LinearRegression(**best_params_length)
    best_model_length.fit(X_train, y_length_train)

    # 最適なモデルの学習（サイズ）
    best_model_size = LinearRegression(**best_params_size)
    best_model_size.fit(X_train, y_size_train)

    print("\nTrainRSME:")
    print(study_length.best_value)
    print(study_size.best_value)

    # モデルの保存
    joblib.dump(best_model_length, './model/linear_model_length.pkl')
    joblib.dump(best_model_size, './model/linear_model_size.pkl')

    return './machine_learning/model/linear_regression_model_length.pkl', './machine_learning/model/linear_regression_model_size.pkl', study_length.best_value, study_size.best_value


def svr_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):

    # OptunaのObjective関数の定義
    def objective_length(trial):
        # ハイパーパラメータの提案
        '''
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'cache_size': trial.suggest_int('cache_size', 100, 300, step=100),
            'coef0': trial.suggest_float('coef0', 0.0, 1.0),
            'degree': trial.suggest_int('degree', 2, 4),
            'epsilon': trial.suggest_float('epsilon', 0.01, 0.5),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        }
        '''
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'cache_size': trial.suggest_int('cache_size', 100, 300, step=100),
            'coef0': trial.suggest_float('coef0', 0.0, 1.0),
            'degree': trial.suggest_int('degree', 2, 4),
            'epsilon': trial.suggest_float('epsilon', 0.01, 0.5),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'max_iter': trial.suggest_categorical('max_iter', [-1, 1000, 10000]),
            'shrinking': trial.suggest_categorical('shrinking', [True, False]),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'verbose': trial.suggest_categorical('verbose', [True, False])
        }
        
        

        model = SVR(**params)
        model.fit(X_train, y_length_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, y_pred, squared=False)

        return rmse

    def objective_size(trial):
        # ハイパーパラメータの提案
        '''
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'cache_size': trial.suggest_int('cache_size', 100, 300, step=100),
            'coef0': trial.suggest_float('coef0', 0.0, 1.0),
            'degree': trial.suggest_int('degree', 2, 4),
            'epsilon': trial.suggest_float('epsilon', 0.01, 0.5),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        }
        '''
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'cache_size': trial.suggest_int('cache_size', 100, 300, step=100),
            'coef0': trial.suggest_float('coef0', 0.0, 1.0),
            'degree': trial.suggest_int('degree', 2, 4),
            'epsilon': trial.suggest_float('epsilon', 0.01, 0.5),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'max_iter': trial.suggest_categorical('max_iter', [-1, 1000, 10000]),
            'shrinking': trial.suggest_categorical('shrinking', [True, False]),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'verbose': trial.suggest_categorical('verbose', [True, False])
        }
        

        model = SVR(**params)
        model.fit(X_train, y_size_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, y_pred, squared=False)

        return rmse

    # Optunaのスタディの作成と最適化の実行
    study_length = optuna.create_study(direction='minimize')
    study_size = optuna.create_study(direction='minimize')
    study_length.optimize(objective_length, n_trials=100)
    study_size.optimize(objective_size, n_trials=100)

    # 最適なハイパーパラメータの表示
    print("Best hyperparameters for length: ", study_length.best_params)
    print("Best hyperparameters for size: ", study_size.best_params)

    # 最適なモデルの定義
    best_model_length = SVR(**study_length.best_params)
    best_model_size = SVR(**study_size.best_params)

    # 最適なモデルの訓練
    best_model_length.fit(X_train, y_length_train)
    best_model_size.fit(X_train, y_size_train)





    print("\nTrainRSME:")
    print(study_length.best_value)
    print(study_size.best_value)

    # モデルの保存
    length_model_path = './model/svr_model_length.pkl'
    size_model_path = './model/svr_model_size.pkl'
    joblib.dump(best_model_length, length_model_path)
    joblib.dump(best_model_size, size_model_path)

    return length_model_path, size_model_path, study_length.best_value, study_size.best_value




def lgbm_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    # ハイパーパラメータ探索の定義
    def objective_length(trial):
        '''
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'max_depth': trial.suggest_int('max_depth', -1, 30),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 30000),
        }
        '''
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'max_depth': trial.suggest_int('max_depth', -1, 30),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 30000),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.7, 1.0, 0.1),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 5),
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.7, 1.0, 0.1),
            'reg_alpha': trial.suggest_categorical('reg_alpha', [0.0, 0.1, 0.5]),
            'reg_lambda': trial.suggest_categorical('reg_lambda', [0.0, 0.1, 0.5]),
            'random_state': 42
        }
        

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_length_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, preds, squared=False)
        return rmse

    def objective_size(trial):
        '''
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'max_depth': trial.suggest_int('max_depth', -1, 30),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 30000),
        }
        '''
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'max_depth': trial.suggest_int('max_depth', -1, 30),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 30000),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.7, 1.0, 0.1),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 5),
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.7, 1.0, 0.1),
            'reg_alpha': trial.suggest_categorical('reg_alpha', [0.0, 0.1, 0.5]),
            'reg_lambda': trial.suggest_categorical('reg_lambda', [0.0, 0.1, 0.5]),
            'random_state': 42
        }
        

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_size_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, preds, squared=False)
        return rmse

    # Optunaによるハイパーパラメータ探索の実行
    study_length = optuna.create_study(direction='minimize')
    study_size = optuna.create_study(direction='minimize')
    study_length.optimize(objective_length, n_trials=100)
    study_size.optimize(objective_size, n_trials=100)

    # 最適なハイパーパラメータの取得
    best_params_length = study_length.best_params
    best_params_size = study_size.best_params

    # 最適なモデルの学習
    best_model_length = lgb.LGBMRegressor(**best_params_length)
    best_model_length.fit(X_train, y_length_train)

    best_model_size = lgb.LGBMRegressor(**best_params_size)
    best_model_size.fit(X_train, y_size_train)


    print("\nTrainRSME:")
    print(study_length.best_value)
    print(study_size.best_value)

    # モデルの保存
    length_model_path = './model/lgbm_model_length.pkl'
    size_model_path = './model/lgbm_model_size.pkl'
    joblib.dump(best_model_length, length_model_path)
    joblib.dump(best_model_size, size_model_path)

    return length_model_path, size_model_path, study_length.best_value, study_size.best_value


def rf_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    
    # Optunaによるハイパーパラメータ探索の定義
    def objective_length(trial):
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        '''
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
        }
        '''
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
            'bootstrap': bootstrap,
            'oob_score': trial.suggest_categorical('oob_score', [True, False]) if bootstrap else False,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': 0,
            'warm_start': trial.suggest_categorical('warm_start', [True, False]),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1),
        }
        

        if bootstrap:
            params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_length_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, preds, squared=False)
        return rmse

    def objective_size(trial):
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        '''
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
        }
        '''
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
            'bootstrap': bootstrap,
            'oob_score': trial.suggest_categorical('oob_score', [True, False]) if bootstrap else False,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': 0,
            'warm_start': trial.suggest_categorical('warm_start', [True, False]),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1),
        }
        

        if bootstrap:
            params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_size_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, preds, squared=False)
        return rmse

    # スタディの設定
    study_length = optuna.create_study(direction='minimize')
    study_size = optuna.create_study(direction='minimize')

    # ハイパーパラメータ探索の実行
    study_length.optimize(objective_length, n_trials=100, timeout=600)
    study_size.optimize(objective_size, n_trials=100, timeout=600)

    # 最適なハイパーパラメータの取得
    best_params_length = study_length.best_params
    best_params_size = study_size.best_params

    # 最適なモデルの学習
    best_model_length = RandomForestRegressor(**best_params_length)
    best_model_size = RandomForestRegressor(**best_params_size)

    best_model_length.fit(X_train, y_length_train)
    best_model_size.fit(X_train, y_size_train)

    print("\nTrainRMSE:")
    print(study_length.best_value)
    print(study_size.best_value)

    # モデルの保存
    length_model_path = './model/rf_model_length.pkl'
    size_model_path = './model/rf_model_size.pkl'
    joblib.dump(best_model_length, length_model_path)
    joblib.dump(best_model_size, size_model_path)

    return length_model_path, size_model_path, study_length.best_value, study_size.best_value



def gb_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    
    # Optunaによるハイパーパラメータ探索の定義
    def objective_length(trial):
        '''
        params = {
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error'])
        }
        '''
        params = {
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
            'alpha': trial.suggest_float('alpha', 0.9, 0.99),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
            'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.2),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 10, 20),
            'tol': trial.suggest_float('tol', 1e-4, 1e-3),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1),
            'random_state': 42,
            'verbose': 0,
            'warm_start': trial.suggest_categorical('warm_start', [True, False]),
            'init': None
        }
        

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_length_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, preds, squared=False)
        return rmse

    def objective_size(trial):
        '''
        params = {
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error'])
        }
        '''
        params = {
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
            'alpha': trial.suggest_float('alpha', 0.9, 0.99),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
            'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.2),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 10, 20),
            'tol': trial.suggest_float('tol', 1e-4, 1e-3),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1),
            'random_state': 42,
            'verbose': 0,
            'warm_start': trial.suggest_categorical('warm_start', [True, False]),
            'init': None
        }
        

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_size_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, preds, squared=False)
        return rmse

    # スタディの設定
    study_length = optuna.create_study(direction='minimize')
    study_size = optuna.create_study(direction='minimize')

    # ハイパーパラメータ探索の実行
    study_length.optimize(objective_length, n_trials=100, timeout=600)
    study_size.optimize(objective_size, n_trials=100, timeout=600)

    # 最適なハイパーパラメータの取得
    best_params_length = study_length.best_params
    best_params_size = study_size.best_params

    # 最適なモデルの学習
    best_model_length = GradientBoostingRegressor(**best_params_length)
    best_model_size = GradientBoostingRegressor(**best_params_size)

    best_model_length.fit(X_train, y_length_train)
    best_model_size.fit(X_train, y_size_train)

    print("\nTrainRSME:")
    print(study_length.best_value)
    print(study_size.best_value)
    
    # モデルの保存
    length_model_path = './model/gb_model_length.pkl'
    size_model_path = './model/gb_model_size.pkl'
    joblib.dump(best_model_length, length_model_path)
    joblib.dump(best_model_size, size_model_path)

    return length_model_path, size_model_path, study_length.best_value, study_size.best_value






def mlp_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    
    # MLP用にさらにスケーリング（必要に応じて）
    #X_train = np.clip(X_train, -1e10, 1e10)  # 極端な値を制限
    #X_val = np.clip(X_val, -1e10, 1e10)
    # OptunaのObjective関数の定義
    def objective_length(trial):
        # ハイパーパラメータの提案
        
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,50), (100,100), (50,100,50)]),
            'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
            'batch_size': trial.suggest_categorical('batch_size', ['auto', 64, 128])
        }
        '''
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,50), (100,100), (50,100,50)]),
            'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
            'batch_size': trial.suggest_categorical('batch_size', ['auto', 64, 128]),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
            'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-1),
            'max_iter': trial.suggest_int('max_iter', 200, 1000),
            'tol': trial.suggest_loguniform('tol', 1e-5, 1e-2),
            'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
            'validation_fraction': trial.suggest_uniform('validation_fraction', 0.1, 0.3),
            'beta_1': trial.suggest_uniform('beta_1', 0.9, 0.999),
            'beta_2': trial.suggest_uniform('beta_2', 0.999, 0.9999),
            'epsilon': trial.suggest_loguniform('epsilon', 1e-8, 1e-6),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 10, 30)
        }
        '''
        
        model = MLPRegressor(**params, random_state=42)
        model.fit(X_train, y_length_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, y_pred, squared=False)
        
        return rmse

    def objective_size(trial):
        # ハイパーパラメータの提案
        
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,50), (100,100), (50,100,50)]),
            'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
            'batch_size': trial.suggest_categorical('batch_size', ['auto', 64, 128])
        }
        '''
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,50), (100,100), (50,100,50)]),
            'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
            'batch_size': trial.suggest_categorical('batch_size', ['auto', 64, 128]),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
            'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-1),
            'max_iter': trial.suggest_int('max_iter', 200, 1000),
            'tol': trial.suggest_loguniform('tol', 1e-5, 1e-2),
            'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
            'validation_fraction': trial.suggest_uniform('validation_fraction', 0.1, 0.3),
            'beta_1': trial.suggest_uniform('beta_1', 0.9, 0.999),
            'beta_2': trial.suggest_uniform('beta_2', 0.999, 0.9999),
            'epsilon': trial.suggest_loguniform('epsilon', 1e-8, 1e-6),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 10, 30)
        }
        '''
        
        model = MLPRegressor(**params, random_state=42)
        model.fit(X_train, y_size_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, y_pred, squared=False)
        
        return rmse

    # Optunaのスタディの作成と最適化の実行
    study_length = optuna.create_study(direction='minimize')
    study_size = optuna.create_study(direction='minimize')
    study_length.optimize(objective_length, n_trials=50)
    study_size.optimize(objective_size, n_trials=50)

    # 最適なハイパーパラメータの取得
    best_params_length = study_length.best_params
    best_params_size = study_size.best_params

    # 最適なモデルの定義
    best_model_length = MLPRegressor(**best_params_length, random_state=42)
    best_model_size = MLPRegressor(**best_params_size, random_state=42)

    # 最適なモデルの訓練
    best_model_length.fit(X_train, y_length_train)
    best_model_size.fit(X_train, y_size_train)
    
    # モデルの保存
    length_model_path = './model/mlp_model_length.pkl'
    size_model_path = './model/mlp_model_size.pkl'
    joblib.dump(best_model_length, length_model_path)
    joblib.dump(best_model_size, size_model_path)

    return length_model_path, size_model_path, study_length.best_value, study_size.best_value