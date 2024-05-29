import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import optuna

# データの読み込み
train_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/train.csv')
val_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/val.csv')
test_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/test.csv')

# 説明変数と目的変数の設定
X_train = train_data[['Sex', 'Age', 'Aneu_neck', 'Aneu_width', 'Aneu_height', 'Aneu_volume', 'Aneu_location', 'Adj_tech', 'Is_bleb']]
y_length_train = train_data['coil_length1']
y_size_train = train_data['coil_size1']

X_val = val_data[['Sex', 'Age', 'Aneu_neck', 'Aneu_width', 'Aneu_height', 'Aneu_volume', 'Aneu_location', 'Adj_tech', 'Is_bleb']]
y_length_val = val_data['coil_length1']
y_size_val = val_data['coil_size1']

X_test = test_data[['Sex', 'Age', 'Aneu_neck', 'Aneu_width', 'Aneu_height', 'Aneu_volume', 'Aneu_location', 'Adj_tech', 'Is_bleb']]
y_length_test = test_data['coil_length1']
y_size_test = test_data['coil_size1']

# カテゴリ変数をダミー変数に変換
X_train = pd.get_dummies(X_train)
X_val = pd.get_dummies(X_val)
X_test = pd.get_dummies(X_test)

# 訓練データとバリデーションデータのカラムを一致させる
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Optunaによるハイパーパラメータ探索の定義
def objective_length(trial):
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
    '''
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_length_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_length_val, preds, squared=False)
    return rmse

def objective_size(trial):
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
    '''
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

# モデルの評価
predictions_length = best_model_length.predict(X_test)
predictions_size = best_model_size.predict(X_test)

rmse_length = mean_squared_error(y_length_test, predictions_length, squared=False)
rmse_size = mean_squared_error(y_size_test, predictions_size, squared=False)

accuracy_length = accuracy_score((y_length_test - predictions_length).abs() <= 5, [True]*len(y_length_test))
accuracy_size = accuracy_score((y_size_test - predictions_size).abs() <= 1, [True]*len(y_size_test))

print("Length Prediction:")
print("RMSE:", rmse_length)
print("Accuracy:", accuracy_length)

print("\nSize Prediction:")
print("RMSE:", rmse_size)
print("Accuracy:", accuracy_size)

print("\nTrainRSME:")
print(study_length.best_value);
print(study_size.best_value);

# モデルの保存
joblib.dump(best_model_length, './machine_learning/model/gb_model_length.pkl')
joblib.dump(best_model_size, './machine_learning/model/gb_model_size.pkl')