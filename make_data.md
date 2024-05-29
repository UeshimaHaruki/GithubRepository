# データの作成手順

## 事前準備

1. excelをcsvに変換
2. ID_relationにID, ID_Med, ID_Patient, ID_adjを登録

## 前処理

make_dataset.pyを動かす

### 前処理の内部で行われていること

- データのクリーニング
    1. 検証したいcsvの読み込み
    2. カラム名や中の日本語を英語に変換
    3. ID_relationから特定のcsvに対して解析用IDを振る
    4. *がついているコイル(欠落コイル)を削除する
    5. データを必要なものに限定する
    6. データをクリーニングする
- 特徴量の追加
    1. coil_countの追加
    2. aneu_width_labelの追加

### 前処理クリーニング6の結果
[まとめたファイル](./for_paper/mkdata_report.csv)


## 事前準備

1. AXIOMから出力したexcelをcsvに変換する(手動)
    - excelの保存先：`./org_excel/axiom_data`
    - csvの保存先：`./org_csv/XXXX.csv`
2. ID_relationにデータを追加する(手動)
    - 保管場所：`./org_csv/ID_relation.csv`
    - ID_relationは`ID, ID_Med, ID_Patient, ID_Aneu`を保持するファイル
    この情報をもとに、各AXIOMから出力したcsvと実験用のcsvを紐づける
        - `ID`：解析用ID, 手動で連番を入力
        - `ID_Med`：病院のID, 慈恵1.順天2
        - `ID_Patient`：患者ID, AXIOMからCopy
        - `ID_Aneu`：動脈瘤ID, AXIOMからCopy

## 前処理

### 動かし方

`python make_dataset.py -path_input 解析用axiomのcsvのパス -path_ID_relation  ID_relationのパス -path_output 保存するファイルのパス`

のコードを実行

### make_dataset.pyの簡単な説明

- データのクリーニング
    1. 解析用axiom.csvの読み込み、ID_relation.csvの読み込み
    2. カラム名と一部の日本語データを英語に変換
    3. 解析用IDの取得
    各axiom.csvとID_relation.csvから情報をマッチングさせてIDを取得。ID以外のカラムは削除される。
    4. 欠落コイルのデータを削除
    5. 特徴量のうち必要なもののみに限定
        
        `list_extract_all = [
            'ID', 'Sex', 'Age',
            'Status_rupture', 'Aneu_type', 
            'Aneu_location','Aneu_location_support',
            'Aneu_neck', 'Aneu_width', 'Aneu_depth', 'Aneu_height','Aneu_volume',
            'Is_retreat', 'Num_adj', 'Adj_tech','Complication',
            'Is_bleb', 'Is_adj_rupture','VER']`
        
    6. メインとなるデータのクリーニング
    
- 特徴量の追加
    1. 使用されたコイルの数を追加(この時、coilの種類は重複しているものも含む)
    カラム名：`coil_count`
    2. 脳動脈瘤の幅のラベルを追加
    カラム名：`aneu_width_label`
    
- デバイスIDの取得
    - 出力ディレクトリの中に自動で`exp`が生成されている。その中に
        
        `used_coil_inc_dup.csv`：重複を含んだ状態でデバイスIDが入力されたもの
        `used_coil.csv`：上記から重複を削除したもの
        
        の2つのファイルが生成される。また、これらのデバイスと紐づいたものが`ID_table`内部に`ID_device.csv`として作られる
        


- ID_relationを必ず確認する事
- 脳動脈瘤の携帯情報と手術技法は欠損値を含んでいる

## メモ

- 合併症の※確認の検証を工藤さんに聞く