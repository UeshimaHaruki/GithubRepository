from src.regression import linear_regression
from src.regression import svr_regression
from src.regression import lgbm_regression
from src.regression import rf_regression
from src.regression import gb_regression
from src.regression import mlp_regression

import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
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
    print(X_train.shape)
    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)
    X_test = pd.get_dummies(X_test)
    print(X_train.shape)
    
    # 訓練データとバリデーションデータのカラムを一致させる
    X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(X_test.shape)
    
    ''''''
    
    # 関数の呼び出し
    linear_length_path, linear_size_path, linear_length_rmse, linear_size_rmse = linear_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    svr_length_path, svr_size_path, svr_length_rmse, svr_size_rmse = svr_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    lgbm_length_path, lgbm_size_path, lgbm_length_rmse, lgbm_size_rmse = lgbm_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    rf_length_path, rf_size_path, rf_length_rmse, rf_size_rmse = rf_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    gb_length_path, gb_size_path, gb_length_rmse, gb_size_rmse = gb_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    mlp_length_path, mlp_size_path, mlp_length_rmse, mlp_size_rmse = mlp_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)

    # 結果の表示
    print("linear")
    print(f"Length model saved at: {linear_length_path}")
    print(f"Size model saved at: {linear_size_path}")
    print(f"Best RMSE for length: {linear_length_rmse}")
    print(f"Best RMSE for size: {linear_size_rmse}")

    print("svr")
    print(f"Length model saved at: {svr_length_path}")
    print(f"Size model saved at: {svr_size_path}")
    print(f"Best RMSE for length: {svr_length_rmse}")
    print(f"Best RMSE for size: {svr_size_rmse}")

    print("lgbm")
    print(f"Length model saved at: {lgbm_length_path}")
    print(f"Size model saved at: {lgbm_size_path}")
    print(f"Best RMSE for length: {lgbm_length_rmse}")
    print(f"Best RMSE for size: {lgbm_size_rmse}")

    print("rf")
    print(f"Length model saved at: {rf_length_path}")
    print(f"Size model saved at: {rf_size_path}")
    print(f"Best RMSE for length: {rf_length_rmse}")
    print(f"Best RMSE for size: {rf_size_rmse}")

    print("gb")
    print(f"Length model saved at: {gb_length_path}")
    print(f"Size model saved at: {gb_size_path}")
    print(f"Best RMSE for length: {gb_length_rmse}")
    print(f"Best RMSE for size: {gb_size_rmse}")

    print("mlp")
    print(f"Length model saved at: {mlp_length_path}")
    print(f"Size model saved at: {mlp_size_path}")
    print(f"Best RMSE for length: {mlp_length_rmse}")
    print(f"Best RMSE for size: {mlp_size_rmse}")

    # Length RMSEの結果をまとめる
    length_results = {
        'Model': ['Linear', 'SVR', 'LGBM', 'Random Forest', 'Gradient Boosting', 'MLP'],
        'Length RMSE': [linear_length_rmse, svr_length_rmse, lgbm_length_rmse, rf_length_rmse, gb_length_rmse, mlp_length_rmse]
    }
    length_results_df = pd.DataFrame(length_results)
    length_results_df.to_csv('./result/length_regression_results.csv', index=False)

    # Size RMSEの結果をまとめる
    size_results = {
        'Model': ['Linear', 'SVR', 'LGBM', 'Random Forest', 'Gradient Boosting', 'MLP'],
        'Size RMSE': [linear_size_rmse, svr_size_rmse, lgbm_size_rmse, rf_size_rmse, gb_size_rmse, mlp_size_rmse]
    }
    size_results_df = pd.DataFrame(size_results)
    size_results_df.to_csv('./result/size_regression_results.csv', index=False)

    # 上位3モデルの選定
    model_paths_and_rmse = [
        (linear_length_path, linear_size_path, linear_length_rmse, linear_size_rmse),
        (svr_length_path, svr_size_path, svr_length_rmse, svr_size_rmse),
        (lgbm_length_path, lgbm_size_path, lgbm_length_rmse, lgbm_size_rmse),
        (rf_length_path, rf_size_path, rf_length_rmse, rf_size_rmse),
        (gb_length_path, gb_size_path, gb_length_rmse, gb_size_rmse),
        (mlp_length_path, mlp_size_path, mlp_length_rmse, mlp_size_rmse)
    ]

    # Length RMSEで上位3モデルを選択
    model_paths_and_rmse.sort(key=lambda x: x[2])  # Length RMSEでソート
    top_3_length_models = model_paths_and_rmse[:3]

    # Size RMSEで上位3モデルを選択
    model_paths_and_rmse.sort(key=lambda x: x[3])  # Size RMSEでソート
    top_3_size_models = model_paths_and_rmse[:3]

    # モデル出力ディレクトリ
    models_dir = './result'
    os.makedirs(models_dir, exist_ok=True)

    # 上位3モデルの予測と評価
    length_evaluation_results = []
    print("Top 3 Length Models Evaluation on Test Data:")
    for rank, (path, _, rmse_length, _) in enumerate(top_3_length_models, start=1):
        model = joblib.load(path)
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_length_test, predictions, squared=False)
        accuracy = accuracy_score((y_length_test - predictions).abs() <= 5, [True]*len(y_length_test))
        length_evaluation_results.append({
            'Model Rank': rank,
            'Model Path': path,
            'RMSE': rmse,
            'Accuracy': accuracy
        })
        print(f"\nModel Rank: {rank}")
        print(f"Model Path: {path}")
        print("RMSE:", rmse)
        print("Accuracy:", accuracy)

        # モデルの保存
        model_name = os.path.join(models_dir, f"length_model_{['first', 'second', 'third'][rank-1]}.pkl")
        joblib.dump(model, model_name)
        print(f"Model saved at: {model_name}")

    length_evaluation_df = pd.DataFrame(length_evaluation_results)
    length_evaluation_df.to_csv('./result/length_evaluation_results.csv', index=False)

    size_evaluation_results = []
    print("\nTop 3 Size Models Evaluation on Test Data:")
    for rank, (_, path, _, rmse_size) in enumerate(top_3_size_models, start=1):
        model = joblib.load(path)
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_size_test, predictions, squared=False)
        accuracy = accuracy_score((y_size_test - predictions).abs() <= 1, [True]*len(y_size_test))
        size_evaluation_results.append({
            'Model Rank': rank,
            'Model Path': path,
            'RMSE': rmse,
            'Accuracy': accuracy
        })
        print(f"\nModel Rank: {rank}")
        print(f"Model Path: {path}")
        print("RMSE:", rmse)
        print("Accuracy:", accuracy)

        # モデルの保存
        model_name = os.path.join(models_dir, f"size_model_{['first', 'second', 'third'][rank-1]}.pkl")
        joblib.dump(model, model_name)
        print(f"Model saved at: {model_name}")

    size_evaluation_df = pd.DataFrame(size_evaluation_results)
    size_evaluation_df.to_csv('./result/size_evaluation_results.csv', index=False)

if __name__ == "__main__":
    main()