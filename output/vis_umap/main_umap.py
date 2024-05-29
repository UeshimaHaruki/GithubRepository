import pandas as pd
import numpy as np
import os
from umap import UMAP
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE


# def mk_pca(data_array, save_dir):
#     is_whiten = False
#     pca = PCA(n_components=2, whiten=is_whiten, random_state=42)
#     pca_result = pca.fit_transform(data_array)
#     # IDとlabelをくっつける
#     ID = [x+1 for x in range(len(pca_result))]
#     pd.DataFrame(np.c_[ID, pca_result], columns=['Exp_ID', 'redu_1', 'redu_2']).to_csv(f'{save_dir}/pca.csv', index=False)


# def mk_tsne(data_array, save_dir, param_grid):
#     for i, perplexity in enumerate(param_grid['perplexity']):
#         for j, early_exaggeration in enumerate(param_grid['early_exaggeration']):
#             tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, random_state=42)
#             tsne_result = tsne.fit_transform(data_array)
#             save_name = f'{save_dir}/tsne_p{perplexity}_ee{early_exaggeration}.csv'

#             ID = [x+1 for x in range(len(tsne_result))]
#             pd.DataFrame(np.c_[ID, tsne_result], columns=['Exp_ID', 'redu_1', 'redu_2']).to_csv(save_name, index=False)


def mk_umap(data_array, save_dir, param_grid):

    for i, n_neighbors in enumerate(param_grid['n_neighbors']):
        for j, min_dist in enumerate(param_grid['min_dist']):
            umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', random_state=42)
            umap_result = umap.fit_transform(data_array)

            save_name = f'{save_dir}/umap_nn{n_neighbors}_md{float(min_dist)*100:.0f}.csv'

            ID = [x+1 for x in range(len(umap_result))]
            pd.DataFrame(np.c_[ID, umap_result], columns=['Exp_ID', 'redu_1', 'redu_2']).to_csv(save_name, index=False)


def main_dim_red(df, save_dir, tsne_param_grid=False, umap_param_grid=False):
    df = df.iloc[:, 1:]
    data_array = pd.get_dummies(df).values
    # mk_pca(data_array, save_dir+'/pca')
    # mk_tsne(data_array, save_dir+'/tsne', tsne_param_grid)
    mk_umap(data_array, save_dir+'/umap', umap_param_grid)
    return True


if __name__=='__main__':
    # mk_dataset()

    # PARAM
    tsne_param_grid = {
        'perplexity': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        'early_exaggeration': [1, 2, 4, 6, 10, 12, 18, 30]}
    # umap_param_grid = {
    #     'n_neighbors': [2, 3, 4, 5, 10, 20, 30, 50, 80],
    #     'min_dist': [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.99]}
    umap_param_grid = {
        'n_neighbors': [3, 5],
        'min_dist': [0.3, 0.8]}
    
    # MAIN
    for filename in ['both', 'medid_1', 'medid_2']:
        print(filename)
        save_root = f'output/{filename}'
        os.makedirs(save_root, exist_ok=True)
        os.makedirs(f'{save_root}/pca', exist_ok=True)
        os.makedirs(f'{save_root}/tsne', exist_ok=True)
        os.makedirs(f'{save_root}/umap', exist_ok=True)
        df = pd.read_csv(f'data/input/{filename}.csv')
        main_dim_red(df, save_root, tsne_param_grid, umap_param_grid)