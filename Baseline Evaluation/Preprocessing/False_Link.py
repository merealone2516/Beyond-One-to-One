import pandas as pd
import numpy as np


file_path = 'path of the false file obtained from Removing_nan.py'
df = pd.read_csv(file_path)


def shuffle_commit_and_file_changes(df):

    grouped_df = df.groupby('Pull_Label')


    shuffled_dfs = []


    for name, group in grouped_df:

        shuffled_group = group.copy()
        shuffled_group['Commit_Message'] = np.random.permutation(group['Commit_Message'].values)
        shuffled_group['File_Changes'] = np.random.permutation(group['File_Changes'].values)

        shuffled_dfs.append(shuffled_group)


    return pd.concat(shuffled_dfs, ignore_index=True)

df_shuffled = shuffle_commit_and_file_changes(df)

df_shuffled.to_csv('path to save', index=False)

print(df_shuffled[['Pull_Label', 'actual_index', 'Commit_Message', 'File_Changes']].head())


