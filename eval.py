import glob
import numpy as np
import pandas as pd

def combine_df(filenames):
    li = []
    for filename in filenames:
        df = pd.read_csv(filename, index_col=0)
        li.append(df)

    combined_df = pd.concat(li, axis=0, ignore_index=True)
    return combined_df

def create_summary_dfs():
    # calculate average metrics by grouping over permutations
    # alcove abstract simulations
    alcove_ab_filenames = glob.glob('csv/alcove_ab/*')
    alcove_ab_df = combine_df(alcove_ab_filenames)
    alcove_ab_group_cols = ['Model', 'Loss Type', 'Image Set', 'LR-Attention', 'LR-Association', 'c', 'phi', 'Type', 'Epoch']
    alcove_ab_summary_df = alcove_ab_df.groupby(alcove_ab_group_cols, as_index=False).mean()
    alcove_ab_summary_df.to_csv('csv/alcove_ab/summary.csv', index=False)
     
    # alcove image simulations
    alcove_im_filenames = glob.glob('csv/alcove_im/*')
    alcove_im_df = combine_df(alcove_im_filenames)
    alcove_im_group_cols = ['Model', 'Net', 'Loss Type', 'Image Set', 'LR-Attention', 'LR-Association', 'c', 'phi', 'Type', 'Epoch']
    alcove_im_summary_df = alcove_im_df.groupby(alcove_im_group_cols, as_index=False).mean()
    alcove_im_summary_df.to_csv('csv/alcove_im/summary.csv', index=False)
     
    # mlp abstract simulations
    mlp_ab_filenames = glob.glob('csv/mlp_ab/*')
    mlp_ab_df = combine_df(mlp_ab_filenames)
    mlp_ab_group_cols = ['Model', 'Loss Type', 'Image Set', 'LR-Association', 'phi', 'Type', 'Epoch']
    mlp_ab_summary_df = mlp_ab_df.groupby(mlp_ab_group_cols, as_index=False).mean()
    mlp_ab_summary_df.to_csv('csv/mlp_ab/summary.csv', index=False)
     
    # mlp image simulations
    mlp_im_filenames = glob.glob('csv/mlp_im/*')
    mlp_im_df = combine_df(mlp_im_filenames)
    mlp_im_group_cols = ['Model', 'Net', 'Loss Type', 'Image Set', 'LR-Association', 'phi', 'Type', 'Epoch']
    mlp_im_summary_df = mlp_im_df.groupby(mlp_im_group_cols, as_index=False).mean()
    mlp_im_summary_df.to_csv('csv/mlp_im/summary.csv', index=False)

if __name__ == "__main__":
    alcove_ab_group_cols = ['Model', 'Loss Type', 'Image Set', 'LR-Attention', 'LR-Association', 'c', 'phi']
    df = pd.read_csv('csv/alcove_ab/summary.csv')
    df_configs = df.drop_duplicates(subset=alcove_ab_group_cols)
    print(df_configs[alcove_ab_group_cols])
    print(len(df_configs[alcove_ab_group_cols]))
