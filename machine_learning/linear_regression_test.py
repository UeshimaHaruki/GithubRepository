import pandas as pd
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

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

# モデルの定義
model_length = LinearRegression()
model_size = LinearRegression()

# ハイパーパラメータ探索の定義（長さ）
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

# ハイパーパラメータ探索の定義（サイズ）
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
joblib.dump(best_model_length, './machine_learning/model/linear_regression_model_length.pkl')
joblib.dump(best_model_size, './machine_learning/model/linear_regression_model_size.pkl')