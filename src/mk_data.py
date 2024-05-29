import pandas as pd
import numpy as np
import os
import math
import sys
sys.path.append('../')
from config.dataset import dict_rename_to_english, list_extract_all


def rename_to_english(df):
    df = df.rename(
        columns=dict_rename_to_english
        )
    df = df.replace({'男':'man', '女':'woman'})
    df = df.replace({'有':1, '無':0})
    return df

def drop_falling_coil(df): # coil_sizeX, lengthX, kindX are str
    for idx in range(len(df)):
        list_size_org = df.loc[idx, 'coil_size1':'coil_size84'].tolist()
        list_length_org = df.loc[idx, 'coil_length1':'coil_length84'].tolist()
        list_kind_org = df.loc[idx, 'coil_kind1':'coil_kind84'].tolist()

        list_size_org = [str(x) if isinstance(x, float) else x for x in list_size_org] # TODO 効率が良い方法
        list_length_org = [str(x) if isinstance(x, float) else x for x in list_length_org] # TODO 効率が良い方法

        # *がついている要素と対応する要素を削除
        filtered_lists = zip(list_size_org, list_length_org, list_kind_org)
        filtered_lists = [(x, y, z) for x, y, z in filtered_lists if x.startswith('*')==False] #  and not x=='nan'] # type(x) == float
        filtered_lists = [(x, y, z) for x, y, z in filtered_lists if x.startswith('●')==False] # 順天用

        filtered_list_size, filtered_list_length, filtered_list_kind = zip(*filtered_lists)
        filtered_list_size = [x for x in filtered_list_size if x!='nan'] # TODO 効率が良い方法
        filtered_list_size = list(filtered_list_size) + [math.nan] * (84 - len(filtered_list_size))
        filtered_list_length = [x for x in filtered_list_length if x!='nan'] # TODO 効率が良い方法
        filtered_list_length = list(filtered_list_length) + [math.nan] * (84 - len(filtered_list_length))
        filtered_list_kind = [x for x in filtered_list_kind if x!='nan'] # TODO 効率が良い方法
        filtered_list_kind = list(filtered_list_kind) + [math.nan] * (84 - len(filtered_list_kind))

        df.loc[idx, 'coil_size1':'coil_size84'] = filtered_list_size
        df.loc[idx, 'coil_length1':'coil_length84'] = filtered_list_length
        df.loc[idx, 'coil_kind1':'coil_kind84'] = filtered_list_kind
    return df

def extract_data(df, list_extract):
    df = df.loc[:,list_extract]
    return df

def cleaning_coil(df): # TODO 変なデータを強引に修正しているので注意
    df = df.reset_index(drop=True)
    for num in range(1,85):
        for idx in range(len(df)):
            if not pd.isnull(df.loc[idx, f'coil_size{num}']):
                if '-' in df.loc[idx, f'coil_size{num}']:
                    df.loc[idx, f'coil_size{num}'] = df.loc[idx, f'coil_size{num}'][0]
                if 'q' in df.loc[idx, f'coil_size{num}']:
                    df.loc[idx, f'coil_size{num}'] = df.loc[idx, f'coil_size{num}'][0]
            if not pd.isnull(df.loc[idx, f'coil_length{num}']):
                if '-' in df.loc[idx, f'coil_length{num}']:
                    df.loc[idx, f'coil_length{num}'] = df.loc[idx, f'coil_length{num}'][0]
    return df
def record_removed_ids(df_before, df_after, step_name, removed_ids):
    # Identify the IDs that have been removed in this step
    removed = df_before.loc[~df_before['ID'].isin(df_after['ID']), 'ID']
    # Create a DataFrame from the removed IDs with the step name
    removed_ids_step = pd.DataFrame(removed)
    removed_ids_step['Removed_Step'] = step_name
    # Concatenate the new removed IDs with the existing ones
    removed_ids = pd.concat([removed_ids, removed_ids_step]).reset_index(drop=True)
    print(removed_ids)
    return removed_ids


def cleaning_data(df_org):
    mod_unrup=True
    mod_VABA=True
    mod_adj_tech=True
    fillna_bleb=True
    del_adj_rupture=True
    get_unrupture=True
    get_no_retreat=True
    get_no_complication=True
    get_aneu_type_saccular=True
    get_location=True
    get_num_adj_one=True
    get_adjunc_tech=True
    dropna_coil_missing=True

    df = df_org.copy()
    list_report = []
    # 変更されたデータフレームを保存する辞書を初期化
    removed_ids = pd.DataFrame(columns=['ID', 'Removed_Step'])

    if mod_unrup: 
        # unrupturedをunruptureに変更
        df = df.replace({'Status_rupture': {'unruptured': 'unrupture'}})
        num_unruptured = len(df_org[df_org['Status_rupture']=='unruptured'])
        list_report.append(['[MOD] unrup', num_unruptured, df.shape, 'modify "Status_rupture" unruptured to unrupture'])
        # unrup Re Treatをunruptureに変更
        df = df.replace({'Status_rupture': {'unrup Re Treat': 'unrupture'}})
        num_unruptured = len(df_org[df_org['Status_rupture']=='unrup Re Treat'])
        list_report.append(['[MOD] unrup', num_unruptured, df.shape, 'modify "Status_rupture" unrup Re Treat to unrupture'])

    if mod_VABA:
        df = df.reset_index(drop=True)
        # VABAをVAかBAに変更
        for idx in range(len(df)):
            if df.loc[idx, 'Aneu_location']=='VABA':
                df.loc[idx, 'Aneu_location'] = df.loc[idx, 'Aneu_location_support']
        num_vaba = len(df_org[df_org['Aneu_location']=='VABA'])
        list_report.append(['[MOD] VABA', num_vaba, df.shape, 'modify "Aneu_location" VABA to VABA'])

    if mod_adj_tech:
        df = df.reset_index(drop=True)
        # Adj_techを修正    
        adj_replace_dict = {
            'Double cathe (DC)': 'Double cathe',
            'Triple cathe.': 'Triple cathe',
            'StentからDC': 'Stent assist',
            'DCからStent': 'Stent assist',
            'BATからStent': 'Stent assist',
            'DCからBAT': 'BAT',
            'BATからDC': 'Double cathe',
            'DC→Stent→DC': 'Stent assist',
            'Rescue Stent': 'Stent assist',
            'SimpleからBAT': 'BAT',
            'DC & Stent': 'Stent assist',
            'StentからBAT': 'Stent assist',
            'SimpleからDC': 'Double cathe',
            'DC & BAT': 'BAT',
            'Stent内からBAT': 'Stent assist',
            'SimpleからStent': 'Stent assist',
            'DC & BAT→Stent': 'Stent assist',
            'BAT→DC→Stent': 'Stent assist',
            'Y Stent+DC': 'Stent assist',
            '記載不可':'Unknown',
            '記入不可':'Unknown'
        }
        df = df.replace({'Adj_tech': adj_replace_dict})
        for original, replacement in adj_replace_dict.items():
            count_replaced = len(df_org[df_org['Adj_tech'] == original])
            list_report.append(['[MOD] ' + original, count_replaced, df.shape, f'modify "Adj_tech" {original} to {replacement}'])
        df_org = df_org.replace({'Adj_tech': adj_replace_dict})

    if fillna_bleb:
        df.Is_bleb = df.Is_bleb.fillna('no')
        num_null = df_org.Is_bleb.isnull().sum()
        list_report.append(['[FILLNA] bleb_data', num_null, df.shape, 'if "bleb_data" is Nan, fillna "no"'])

    if get_unrupture:
        df = df.reset_index(drop=True)
        df_before = df.copy()
        # ruptureデータを削除
        df = df[df['Status_rupture']=='unrupture']
        num_drop_rupture = len(df_org[df_org['Status_rupture']=='rupture'])
        list_report.append(['[GET] rupture', num_drop_rupture, df.shape, 'del "Status_ruptrue" is rupture'])
        removed_ids = record_removed_ids(df_before, df, 'rupture', removed_ids)

    if del_adj_rupture:
        df = df.reset_index(drop=True)
        df_before = df.copy()
        # 術中破裂があったデータは削除
        df = df[df['Is_adj_rupture'] != 'yes']
        num_no_retreat = len(df_org[df_org['Is_adj_rupture'] == 'yes'])
        list_report.append(['[DEL] adj_rupture', num_no_retreat, df.shape, 'del "Is_adj_rupture" not one'])
        removed_ids = record_removed_ids(df_before, df, 'adj_rupture', removed_ids)

    if get_no_retreat:
        df = df.reset_index(drop=True)
        df_before = df.copy()
        # retreatが1にデータを限定
        df = df[df['Is_retreat']==0]
        num_no_retreat = len(df_org[df_org['Is_retreat'] != 0])
        list_report.append(['[GET] no_retreat', num_no_retreat, df.shape, 'del "Is_retreat" not one'])
        removed_ids = record_removed_ids(df_before, df, 'retreat', removed_ids)

    if get_no_complication:
        df = df.reset_index(drop=True)
        df_before = df.copy()
        # 合併症が無いデータに限定
        # Is_del_complicationが1のデータは無条件に削除
        df = df[df['Is_del_complication'] != 1] # TODO あまり適切な方法ではない
        df = df.reset_index(drop=True)
        kome_bool = df['Complication'].str.startswith('※')
        check_bool = df['Complication'].str.startswith('確認')
        for i in range(len(df)):
            if kome_bool[i] == True:
                df.loc[i, 'Complication'] = 'none'
            elif check_bool[i] == True:
                df.loc[i, 'Complication'] = 'none'
        df.Complication = df.Complication.fillna('none')
        df = df[df.Complication == 'none']
        # 合併症が確認されてないデータに限定
        num_del_complication = len(df_org[df_org['Is_del_complication'] == 1])
        kome_bool = df_org['Complication'].str.startswith('※')
        check_bool = df_org['Complication'].str.startswith('確認')
        for i in range(len(df_org)):
            if kome_bool[i] == True:
                df_org.loc[i, 'Complication'] = 'none'
            elif check_bool[i] == True:
                df_org.loc[i, 'Complication'] = 'none'
        df_org.Complication = df_org.Complication.fillna('none')
        num_null = len(df_org[df_org.Complication != 'none']) - num_del_complication
        list_report.append(['[GET] no_complication', num_null, df.shape, 'if "Complication" is Nan_*_"KAKUNIN"_attempt, fillna "none"'])
        removed_ids = record_removed_ids(df_before, df, 'complication', removed_ids)

    if get_aneu_type_saccular:
        df = df.reset_index(drop=True)
        df_before = df.copy()
        # sacular以外のデータを削除
        df = df[df['Aneu_type'] == 'saccular']
        num_not_saccular = len(df_org[df_org['Aneu_type'] != 'saccular'])
        list_report.append(['[GET] aneu type == saccular', num_not_saccular, df.shape, 'del "Aneu_type" not saccular'])
        removed_ids = record_removed_ids(df_before, df, 'aneu_type', removed_ids)

    if get_location:
        df = df.reset_index(drop=True)
        df_before = df.copy()
        # ICA, VA, BA, ACA, MCAにデータを限定
        list_location = [
            'ICA','VA','BA', 'ACA', 'MCA'
        ]
        for index, data in df.iterrows():
            bool_location = data['Aneu_location'] in list_location
            if bool_location == False:
                df = df.drop(index = index) 
        # ICA, VA, BA, ACA, MCA以外のデータ数をカウント
        num_del = 0 
        list_location.append('VABA') # df_orgにはVABAで記録されているのでVABAを追加してカウントする
        for index, data in df_org.iterrows():
            bool_location = data['Aneu_location'] in list_location
            if bool_location == False:
                num_del += 1  
        list_report.append(['[GET] location', num_del, df.shape, 'del "Aneu_location" are not ICA,VA,BA,ACA,MCA'])
        removed_ids = record_removed_ids(df_before, df, 'location', removed_ids)

    if get_num_adj_one:
        df = df.reset_index(drop=True)
        df_before = df.copy()
        # Num_adjが1にデータを限定
        df = df[df['Num_adj'] == 1]
        num_adj_one = len(df_org[df_org['Num_adj'] != 1])
        list_report.append(['[GET] num_of_treat_one', num_adj_one, df.shape, 'del "Num_adj" not one'])
        removed_ids = record_removed_ids(df_before, df, 'num_adj', removed_ids)
        
    if get_adjunc_tech:
        df_before = df.copy()
        adj_list = [
            'Simple',
            'Double cathe',
            'Triple cathe',
            'BAT',
            'Stent assist', 
            'Unknown'
            ]
        df['Adj_tech'] = df['Adj_tech'].fillna('Unknown') # TODO 欠損値をUnknownにするのは良くない
        for index, data in df.iterrows():
            bool_adj = data['Adj_tech'] in adj_list
            if bool_adj == False:
                df = df.drop(index = index) 
        df = df.replace({'Adj_tech': {'Unknown': np.nan}}) # TODO Unknownを欠損値にするのは良くない
        # count how many data will be removed 
        num_del = 0 
        for index, data in df_org.iterrows():
            bool_adj = data['Adj_tech'] in adj_list
            if bool_adj == False:
                num_del += 1  
        list_report.append(['[GET] Adj_tech', num_del, df.shape, 'If "Adj_tech" is not in "Simple","Double cathe","Triple cathe","BAT","Stent assist","Unknown", the cases are removed.'])
        removed_ids = record_removed_ids(df_before, df, 'adj_tech', removed_ids)

    if dropna_coil_missing:
        df = df.reset_index(drop=True)
        df_before = df.copy()
        # coil_size1, length1が欠損しているデータを削除
        df = df.dropna(subset=['coil_size1', 'coil_length1'])
        num_size_missing = df_org['coil_size1'].isnull().sum()
        num_length_missing = df_org['coil_length1'].isnull().sum()
        list_report.append(['[DROPNA] coil_size1_missing', num_size_missing, df.shape, 'df.shape is same coil_size1/length1'])
        list_report.append(['[DROPNA] coil_length1_missing', num_length_missing, df.shape, 'df.shape is same coil_size1/length1'])
        removed_ids = record_removed_ids(df_before, df, 'coil_missing', removed_ids)
    
    # coilのサイズや長さの異常値を修正
    df = cleaning_coil(df)

    # Status_rupture, Aneu_type, Aneu_location_support, Is_retreat, Num_adj, Complication, Is_adj_ruptureを削除
    df = df.drop(['Status_rupture', 'Aneu_type', 'Aneu_location_support', 'Is_retreat', 'Num_adj', 'Complication','Is_del_complication', 'Is_adj_rupture'], axis=1)
    df = df.reset_index(drop=True)
    list_report.append(['[DROP_COL]', 0, df.shape, 'Drop columns "Status_rupture", "Aneu_type", "Aneu_location_support", "Is_retreat", "Num_adj", "Complication","Is_del_complication", "Is_adj_rupture"'])

    df_report = pd.DataFrame(list_report)
    return df, df_report, removed_ids

def coil_counter(df) -> list:
    """
    df: pd.DataFrame
    return:list
    """
    list_count = []
    for idx in range(len(df)):
        for x in range(1,85):
            if x==84:
                list_count.append(x)
                break
            elif pd.isnull(df.loc[idx, f'coil_size{x+1}']):
                list_count.append(x)
                break
            else:
                pass
    return list_count

def set_aneu_height_label(df) -> pd.DataFrame:
    ### Aneu_heightが5未満,5以上7未満,7以上のラベルを付与
    df['Aneu_height_label'] = np.nan
    df.loc[df['Aneu_height']<5, 'Aneu_height_label'] = 0
    df.loc[(df['Aneu_height']>=5) & (df['Aneu_height']<7), 'Aneu_height_label'] = 1
    df.loc[df['Aneu_height']>=7, 'Aneu_height_label'] = 2
    pd.DataFrame([[0, 'height<5'], [1, '5<=height<7'], [2, '7<=height']], columns=['Aneu_height_ID', 'aneu_height_label_name']).to_csv('./dataset/ID_Aneu_height.csv', index=False)
    return df


    # drop_aneu_stats=True
    # drop_aneu_volume=True
    # tobe_froat_size=True

    # if drop_aneu_stats:
    #     df = df.dropna(subset=['Aneu_neck', 'Aneu_width', 'Aneu_depth', 'Aneu_height', 'Aneu_volume'])
    #     num_dropna_neck = df_org['Aneu_neck'].isnull().sum()
    #     num_dropna_width = df_org['Aneu_width'].isnull().sum()
    #     num_dropna_depth = df_org['Aneu_depth'].isnull().sum()
    #     num_dropna_height = df_org['Aneu_height'].isnull().sum()
    #     num_dropna_volume = df_org['Aneu_volume'].isnull().sum()
    #     list_report.append(['[DROPNA] Aneu_neck', num_dropna_neck, df.shape, 'df.shape is same neck, width, depth, height, volume'])
    #     list_report.append(['[DROPNA] Aneu_width', num_dropna_width, df.shape, 'df.shape is same neck, width, depth, height, volume'])
    #     list_report.append(['[DROPNA] Aneu_depth', num_dropna_depth, df.shape, 'df.shape is same neck, width, depth, height, volume'])
    #     list_report.append(['[DROPNA] Aneu_height', num_dropna_height, df.shape, 'df.shape is same neck, width, depth, height, volume'])
    #     list_report.append(['[DROPNA] Aneu_volume', num_dropna_volume, df.shape, 'df.shape is same neck, width, depth, height, volume'])

    # if drop_aneu_volume:
    #     df = df[df['Aneu_volume'] != 0]
    #     df = df.dropna(subset = ['Aneu_volume'])
    #     num_dropna = len(df_org[df_org['Aneu_volume'] == 0])
    #     list_report.append(['[DROPNA] aneurysm_volume_iszaro', num_dropna, df.shape, 'If "Aneu_volume" is 0, drop the data.'])


    # def identify_id(df, df_ID_relation):
#     df = pd.merge(df, df_ID_relation, on=['ID_Patient', 'ID_Adj'])
#     return df