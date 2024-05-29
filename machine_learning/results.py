import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# テストデータの読み込み
test_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/test.csv')

# 説明変数と目的変数の設定
X_test = test_data[['Sex', 'Age', 'Aneu_neck', 'Aneu_width', 'Aneu_height', 'Aneu_volume', 'Aneu_location', 'Adj_tech', 'Is_bleb']]
y_length_test = test_data['coil_length1']
y_size_test = test_data['coil_size1']

# カテゴリ変数をダミー変数に変換
X_test = pd.get_dummies(X_test)

# モデルファイルのリストと名前の対応
model_files = {
    'mlp_length': './machine_learning/model/mlp_model_length.pkl',
    'mlp_size': './machine_learning/model/mlp_model_size.pkl',
    'rf_length': './machine_learning/model/rf_model_length.pkl',
    'rf_size': './machine_learning/model/rf_model_size.pkl',
    'gb_length': './machine_learning/model/gb_model_length.pkl',
    'gb_size': './machine_learning/model/gb_model_size.pkl',
    'lgbm_length': './machine_learning/model/lgbm_model_length.pkl',
    'lgbm_size': './machine_learning/model/lgbm_model_size.pkl',
    'svr_length': './machine_learning/model/svr_model_length.pkl',
    'svr_size': './machine_learning/model/svr_model_size.pkl',
    'linear_length': './machine_learning/model/linear_regression_model_length.pkl',
    'linear_size': './machine_learning/model/linear_regression_model_size.pkl'
}

# 結果を格納するDataFrameを作成
results = []

# 各モデルごとに処理を行う
for model_name, model_file in model_files.items():
    # モデルの読み込み
    model = joblib.load(model_file)
    
    # モデルの予測
    predictions = model.predict(X_test)
    
    # RMSEの計算
    if 'length' in model_name:
        y_test = y_length_test
        accuracy = accuracy_score((y_test - predictions).abs() <= 5, [True]*len(y_test))
    else:
        y_test = y_size_test
        accuracy = accuracy_score((y_test - predictions).abs() <= 1, [True]*len(y_test))
    
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    # 結果をリストに追加
    result = {'Model': model_name, 'RMSE': rmse, 'Accuracy': accuracy}
    results.append(result)

# 結果をDataFrameに変換
results_df = pd.DataFrame(results)

# 結果をCSVファイルに出力
results_df.to_csv('./model_evaluation_results.csv', index=False)

# 棒グラフを作成
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# SizeモデルのRMSEをプロット
size_results = results_df[results_df['Model'].str.contains('size')].sort_values(by='RMSE')
axes[0].barh(size_results['Model'], size_results['RMSE'], color='skyblue')
axes[0].set_title('RMSE for Size Models')
axes[0].set_xlabel('RMSE')
axes[0].set_ylabel('Model')

# LengthモデルのRMSEをプロット
length_results = results_df[results_df['Model'].str.contains('length')].sort_values(by='RMSE')
axes[1].barh(length_results['Model'], length_results['RMSE'], color='lightgreen')
axes[1].set_title('RMSE for Length Models')
axes[1].set_xlabel('RMSE')
axes[1].set_ylabel('Model')

plt.tight_layout()
plt.savefig('./rmse_comparison.png')
plt.show()

# 最も正答率が高いサイズモデルと長さモデルの予測を取得
best_size_model = results_df[results_df['Model'].str.contains('size')].sort_values(by='Accuracy', ascending=False).iloc[0]['Model']
best_length_model = results_df[results_df['Model'].str.contains('length')].sort_values(by='Accuracy', ascending=False).iloc[0]['Model']
best_size_predictions = joblib.load(model_files[best_size_model]).predict(X_test)
best_length_predictions = joblib.load(model_files[best_length_model]).predict(X_test)

# プロット
plt.figure(figsize=(10, 6))

# サイズの散布図
plt.figure(figsize=(10, 6))
plt.scatter(best_size_predictions, y_size_test, color='blue', label='Predicted Size')
plt.plot(y_size_test, y_size_test, color='red', linestyle='-', label='Ideal Prediction Line')
plt.plot(y_size_test+1, y_size_test, color='red', linestyle='--', label='Ideal Prediction Line')
plt.plot(y_size_test-1, y_size_test, color='red', linestyle='--', label='Ideal Prediction Line')
plt.xlabel('Predicted Size')
plt.ylabel('Actual Size')
plt.title('Predicted Size vs Actual Size')
plt.legend()
#plt.xlim(0, 20)  # X軸の範囲を0から20に設定
#plt.ylim(0, 20)  # Y軸の範囲を0から40に設定
plt.grid(True)
plt.savefig('./predicted_size.png')
plt.show()

# 長さの散布図
plt.figure(figsize=(10, 6))
plt.scatter(best_length_predictions, y_length_test, color='green', label='Predicted Length')
plt.plot(y_length_test, y_length_test, color='green', linestyle='--', label='Ideal Prediction Line')
plt.plot(y_length_test+5, y_length_test, color='green', linestyle='--', label='Ideal Prediction Line')
plt.plot(y_length_test-5, y_length_test, color='green', linestyle='--', label='Ideal Prediction Line')
plt.xlabel('Predicted Length')
plt.ylabel('Actual Length')
plt.title('Predicted Length vs Actual Length')
plt.legend()
#plt.xlim(0, 45)  # X軸の範囲を0から20に設定
#plt.ylim(0, 45)  # Y軸の範囲を0から40に設定
plt.grid(True)
plt.savefig('./predicted_length.png')
plt.show()