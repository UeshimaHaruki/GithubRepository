import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


def mkfig(result, label:list, palette:dict, save_name:str)->None:
    """save figure

    Args:
        result (_type_): result
        label (list): label list
        palette (dict): palette dict
        save_name (str): path of save figure
    """

    sort_index = np.argsort(label)
    result_sorted = result.iloc[sort_index].reset_index(drop=True)
    label_sorted = np.array(label)[sort_index]
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    sns.scatterplot(x=result_sorted.loc[:, 'redu_1'], y=result_sorted.loc[:, 'redu_2'], hue=label_sorted, palette=palette, legend="full", ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(loc='upper right', fontsize=15)
    # plt.show()
    plt.tight_layout()
    fig.savefig(save_name)
    plt.close(fig)

# ROOT = 'samp@le'

# output_dirname ='0818_dimention_reduction_grayscale'
# option='point'

# df_palette = pd.read_csv(f'{ROOT}/label_color_mapping.csv')
# for step in ['step1', 'step2', 'step3']:
#     save_root = f'{ROOT}/{output_dirname}/{step}'
#     mk_dir(save_root)

#     for model in ['pca', 'tsne', 'umap']:
#         listdirs = os.listdir(f'{ROOT}/0809_dimention_reduction/{step}/{model}')
#         for listdir in listdirs:
#             df = pd.read_csv(f'{ROOT}/0809_dimention_reduction/{step}/{model}/{listdir}')
#             result = df.loc[:, 'redu_1':'redu_2']
#             label_palette = df_palette.set_index(f'label_{option}_{step}')[f'palette_{option}_{step}'].to_dict()
#             label_palette = {k: ast.literal_eval(v) for k, v in label_palette.items()}
#             mkfig(result, df_palette[f'label_{option}_{step}'], label_palette, os.path.join(save_root, model, listdir.replace('csv', 'png')))


def mkfig_tmp(result, label:list, save_name:str)->None:
    """save figure

    Args:
        result (_type_): result
        label (list): label list
        palette (dict): palette dict
        save_name (str): path of save figure
    """

    cmap = mcolors.LinearSegmentedColormap.from_list("reds", ["white", "red"])

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    sns.scatterplot(x=result.loc[:, 'redu_1'], y=result.loc[:, 'redu_2'], hue=label, palette=cmap, ax=ax) #, legend="full")
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.legend(loc='upper right', fontsize=15)
    # plt.show()
    plt.tight_layout()
    fig.savefig(save_name)
    plt.close(fig)


if __name__=='__main__':
    for data_name in ['both', 'medid_1', 'medid_2']:
    # data_name = 'medid_1'
        print(data_name)
        os.makedirs(f'mapping/{data_name}', exist_ok=True)
        save_root = f'mapping/{data_name}'

        os.makedirs(save_root, exist_ok=True)
        os.makedirs(f'{save_root}/pca', exist_ok=True)
        os.makedirs(f'{save_root}/tsne', exist_ok=True)
        os.makedirs(f'{save_root}/umap', exist_ok=True)

        # labelがあるcsvを読み込み
        df_label = pd.read_csv(f'data/label/{data_name}.csv', encoding='ISO-8859-1')
        for col_label in ['Age', 'Aneu_width', 'Aneu_volume', 'coil_length1', 'coil_size1', 'coil_count']:
            label = df_label[col_label]

            # resultを読み込む
            for filename in os.listdir(f'output/{data_name}/umap'):
                df_umap = pd.read_csv(f'output/{data_name}/umap/{filename}')
                result = df_umap.loc[:, 'redu_1':'redu_2']
                mkfig_tmp(result, label, f'{save_root}/umap/{filename.replace(".csv", f"_{col_label}.png")}')

        if data_name=='both':
            label = df_label['med_id']
            for filename in os.listdir(f'output/{data_name}/umap'):
                df_umap = pd.read_csv(f'output/{data_name}/umap/{filename}')
                result = df_umap.loc[:, 'redu_1':'redu_2']
                mkfig_tmp(result, label, f'{save_root}/umap/{filename.replace(".csv", f"_med_id.png")}')

