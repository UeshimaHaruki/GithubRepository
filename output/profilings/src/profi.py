import pandas as pd
import pandas_profiling as pdp
import os



"""
使い方
・手動で./inputフォルダにファイルを入れる
・必要な名前と中身に編集する
9月15日(金)：experimentは解析直前のファイル．axiomは元のファイル

"""
list_dir = os.listdir('../profilings_input')

if __name__ =='__main__':
    for dir_name in list_dir:
        file_name, _ = dir_name.split('.')
        df = pd.read_csv('../profilings_input/'+file_name+'.csv')
        profile = pdp.ProfileReport(df)
        profile.to_file('../profilings_result/'+ file_name +'.html')