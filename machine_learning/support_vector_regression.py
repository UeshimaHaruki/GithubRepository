import optuna
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# データの読み込み
train_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/train.csv')
test_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/test.csv')
val_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/val.csv')

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

# 標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

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
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'max_iter': trial.suggest_categorical('max_iter', [-1, 1000, 10000]),
        'shrinking': trial.suggest_categorical('shrinking', [True, False]),
        'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
        'verbose': trial.suggest_categorical('verbose', [True, False])
    }
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
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'max_iter': trial.suggest_categorical('max_iter', [-1, 1000, 10000]),
        'shrinking': trial.suggest_categorical('shrinking', [True, False]),
        'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
        'verbose': trial.suggest_categorical('verbose', [True, False])
    }
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
joblib.dump(best_model_length, './machine_learning/model/svr_model_length.pkl')
joblib.dump(best_model_size, './machine_learning/model/svr_model_size.pkl')