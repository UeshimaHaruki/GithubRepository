import pandas as pd
import numpy as np

def remove_duplicate(df):
    # 'ID'列以外の列名を取得
    coil_cols = [col for col in df.columns if col != 'ID']

    # 'ID'列を基準にループ
    for idx, row in df.iterrows():
        # 重複を削除し、NaN埋
        unique_coils = pd.Series(row[coil_cols]).drop_duplicates().tolist()
        unique_coils += [np.nan] * (len(coil_cols) - len(unique_coils))
        # 更新
        df.loc[idx, coil_cols] = unique_coils
    return df

def mk_device_table(df):
    list_used_coil = []
    # コイルのサイズと長さからデバイス名を取得する．
    for idx in range(len(df)):
        ID = df.loc[idx, 'ID']
        list_patient_coil = [ID]
        for j in range(1,85):
            # 変なsize, lengthが入っているので例外処理
            try:
                size = float(df.loc[idx, f'coil_size{j}'])
                length = float(df.loc[idx, f'coil_length{j}'])
            except:
                print(f"Some thing is wrong {df.loc[idx, f'coil_size{j}'], df.loc[idx, f'coil_length{j}']}")
            #　欠損値を迎えた時点で解析終了
            if np.isnan(size) or np.isnan(length):
                break
            else:
                list_patient_coil.append(f'size{int(size*10):03}_length{int(length*10):03}')
        # list length0を防ぐ
        if len(list_patient_coil)>0:
            list_used_coil.append(list_patient_coil)
    # DataFrameに変換
    max_length = max([len(i) for i in list_used_coil])
    df_used_coil = pd.DataFrame(list_used_coil, columns=['ID']+['coil'+str(i) for i in range(1,max_length)])

    #　使用されたすべてのデバイスに新しくIDを振る
    list_device = []
    for idx in range(len(df_used_coil)):
        for j in range(1, df_used_coil.shape[1]):
            device = df_used_coil.loc[idx, f'coil{j}']
            if device not in list_device:
                list_device.append(device)
    # Noneを削除+Sort
    list_device = [x for x in list_device if x is not None]
    list_device.sort()
    # IDを振って保存
    df_device = pd.DataFrame(list_device, columns=['Device'])
    df_device['Device_ID'] = [i+1 for i in range(len(df_device))]

    # デバイス名からデバイスIDに変更する
    for idx in range(len(df_used_coil)):
        for j in range(1, df_used_coil.shape[1]):
            device = df_used_coil.loc[idx, f'coil{j}']
            matched_rows = df_device.loc[df_device['Device']==device, 'Device_ID']
            if not matched_rows.empty:
                df_used_coil.loc[idx, f'coil{j}'] = matched_rows.iloc[0]

    # 各行の重複を削除
    df_used_coil_no_dup = remove_duplicate(df_used_coil)
    return df_device, df_used_coil, df_used_coil_no_dup
