import pandas as pd
import numpy as np
import os
import math
import argparse
from sklearn.model_selection import train_test_split
from scipy.stats import zscore


def fillna_morphology(df, list_report):
    """
    処理の内容
    1. Aneu_height, Aneu_width, Aneu_neck, Depthのうち2個以上欠損がある場合はデータを削除します
    2. Widthのみが欠損：Neckで補完
    3. Neckのみが欠損：Widthで補完
    4. Depthが欠損：Widthの値で補完
    5. Volumeが欠損or0の場合は、Height * Aneu_width * Aneu_neck * 0.523で補完
    6. Depthを特徴量から削除
    """
    df = df.copy()
    num_before_drop = len(df)
    df = df.dropna(subset=['Aneu_height', 'Aneu_width', 'Aneu_neck', 'Aneu_depth'], thresh=2)
    # レポートの作成
    num_after_drop = len(df)
    list_report.append(["Number of dropped data (2 morph. missing)", num_before_drop - num_after_drop, df.shape])
    list_report.append(['Number of fillna width', df['Aneu_width'].isnull().sum(), df.shape])
    list_report.append(['Number of fillna neck', df['Aneu_neck'].isnull().sum(), df.shape])
    list_report.append(['Number of fillna depth', df['Aneu_depth'].isnull().sum(), df.shape])
    list_report.append(['Number of fillna volume', df['Aneu_volume'].isnull().sum(), df.shape])
    list_report.append(['Number of zero volume', len(df[df['Aneu_volume']==0]), df.shape])
    # 処理の実行
    df['Aneu_width'] = df['Aneu_width'].fillna(df['Aneu_neck'])
    df['Aneu_neck'] = df['Aneu_neck'].fillna(df['Aneu_width'])
    # df['Aneu_height'] = df['Aneu_height'].fillna((df['Aneu_width'] + df['Aneu_neck']) / 2) # TODO 欠損補完方法に難あり+事例が存在しない
    df['Aneu_depth'] = df['Aneu_depth'].fillna((df['Aneu_width'] + df['Aneu_neck'] + df['Aneu_height']) / 3) 
    # Aneu_valume=0をnp.nanに
    df['Aneu_volume'] = df['Aneu_volume'].replace(0, np.nan)
    df['Aneu_volume'] = df['Aneu_volume'].fillna(df['Aneu_height'] * df['Aneu_width'] * df['Aneu_depth'] * math.pi / 6)
    df = df.drop('Aneu_depth', axis=1)
    list_report.append(['Final shape(drop aneu. depth)', '---', df.shape])
    return df, list_report

def set_aneu_height_label(df, list_report):
    # レポート作成
    list_report.append(['Number of Aneu_height_label nan', df['Aneu_height_label'].isnull().sum(), df.shape])
    
    # Aneu_widthが5未満,5以上7未満,7以上のラベルを付与
    df['Aneu_height_label'] = np.nan
    # ラベル付与．実装
    df.loc[df['Aneu_height']<5, 'Aneu_height_label'] = 0
    df.loc[(df['Aneu_height']>=5) & (df['Aneu_height']<7), 'Aneu_height_label'] = 1
    df.loc[(df['Aneu_height']>=7) & (df['Aneu_height']<10), 'Aneu_height_label'] = 2
    df.loc[df['Aneu_height']>=10, 'Aneu_height_label'] = 3
    
    # レポート作成
    list_report.append([
        'Number of new aneu width label (0 / 1 / 2)',
        f'{len(df[df["Aneu_height_label"]==0])} / {len(df[df["Aneu_height_label"]==1])} / {len(df[df["Aneu_height_label"]==2])} / {len(df[df["Aneu_height_label"]==3])}',
        df.shape]) 
    
    return df, list_report

def rm_aneu_height_label_3(df, list_report):
    """aneu_height_label==3のデータを削除"""
    num_label_3 = len(df[df['Aneu_height_label']==3])
    df = df[df['Aneu_height_label']!=3]
    df = df.reset_index(drop=True)

    # レポート作成
    list_report.append(['Final shape(drop aneu height label 3)', num_label_3, df.shape])
    return df, list_report

def del_data_outlier(df, list_report):
    """
    外れ値の削除
        Args:
            df: DataFrame
        Returns:
            df: DataFrame
    """
    columns_to_check =  ['Aneu_neck', 'Aneu_width', 'Aneu_height', 'Aneu_volume']

    threshold = 4
    for col_name in columns_to_check:
        # TODO IQRによる外れ値の削除は一旦Stay
        # Q1 = df[col_name].quantile(0.25)
        # Q3 = df[col_name].quantile(0.75)
        # IQR = Q3 - Q1

        # filter = (df[col_name] >= Q1 - 1.5 * IQR) & (df[col_name] <= Q3 + 1.5 *IQR)
        # df = df[filter]

        ## ZScrreによる外れ値の削除
        zscores = zscore(df[col_name])
        # zscoresが閾値を超えたデータのcol_nameの値とIDを出力
        # print(df[(zscores <= -threshold) | (zscores >= threshold)][['ID', col_name]])

        # レポート作成
        list_report.append([
            f'Number of outlier : {col_name} \n {df[(zscores <= -threshold) | (zscores >= threshold)][["ID", col_name]]} \n [Attention] 徐々に削れていることに注意', 
            len(df[(zscores <= -threshold) | (zscores >= threshold)]), 
            df.shape])
        df = df[(zscores > -threshold) & (zscores < threshold)]
    
    list_report.append(['Final shape(drop outlier)', '---', df.shape])
    df = df.reset_index(drop=True)
    return df, list_report

def split_data(df):
    """
    処理の内容
    1. データを学習用とテスト用に分割します
    2. データのIDを決定します
    """
    df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Aneu_height_label'])
    df_train, df_val = train_test_split(df_train_val, test_size=(1/8), random_state=42, stratify=df_train_val['Aneu_height_label'])
    df_train = df_train.sort_values('ID').reset_index(drop=True)
    df_val = df_val.sort_values('ID').reset_index(drop=True)
    df_test = df_test.sort_values('ID').reset_index(drop=True)
    df_train_ID = df_train[['ID', 'Aneu_height_label']]
    df_val_ID = df_val[['ID', 'Aneu_height_label']]
    df_test_ID = df_test[['ID', 'Aneu_height_label']]
    df_train_ID.loc[:, 'Split_Category'] = 'Train'
    df_val_ID.loc[:, 'Split_Category'] = 'Val'
    df_test_ID.loc[:, 'Split_Category'] = 'Test'
    df_ID_split_category = pd.concat([df_train_ID, df_val_ID, df_test_ID], axis=0).sort_values('ID').reset_index(drop=True)
    return df_train, df_val, df_test, df_ID_split_category

def main(path_input, path_outputdir):
    """
    処理の内容
    1. 欠損値補完
    2. one-hot encoding
    3. データの分割とIDの決定
    """
    list_report = []
    # データの読み込み
    df = pd.read_csv(path_input)
    # 欠損値補完
    df, list_report = fillna_morphology(df, list_report)
    # Aneu_widthラベルの振り直し
    df, list_report = set_aneu_height_label(df, list_report)
    # Aneu_height_label==3のデータを削除
    df, list_report = rm_aneu_height_label_3(df, list_report)
    # 外れ値の削除
    df, list_report = del_data_outlier(df, list_report)
    print(df.shape)
    # データの分割とIDの決定
    df_train, df_val, df_test, df_ID_split_category = split_data(df)
    # ディレクトリの作成
    os.makedirs(path_outputdir, exist_ok=True)
    # レポート保存
    pd.DataFrame(list_report).to_csv(f'{path_outputdir}/report.csv', index=False)
    # 各データセットの保存
    df_train.to_csv(f'{path_outputdir}/train.csv', index=False)
    df_val.to_csv(f'{path_outputdir}/val.csv', index=False)
    df_test.to_csv(f'{path_outputdir}/test.csv', index=False)
    df = pd.concat([df_train, df_val, df_test], axis=0).sort_values('ID').reset_index(drop=True)
    df.to_csv(f'{path_outputdir}/dataset.csv', index=False)
    df_ID_split_category.to_csv(f'{path_outputdir}/ID_split_category.csv', index=False)
    print('Preprocess is done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_input', type=str, default='./output/medid_1_v8/experiment.csv')
    parser.add_argument('-path_outputdir', type=str, default='./output/exp_dataset/medid_1_v8')
    args = parser.parse_args()
    path_input = args.path_input
    path_outputdir = args.path_outputdir
    main(path_input, path_outputdir)