import pandas as pd
import numpy as np
import os


dataset_paths = [ " Path of the datasets  "  ]

save_path = " Path to save"

os.makedirs(save_path, exist_ok=True)


for file_path in dataset_paths:

    df = pd.read_csv(file_path)


    dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    label_0_df = df[df['label'] == 0]
    label_1_df = df[df['label'] == 1]

    unique_pull_labels_0 = label_0_df['Pull_Label'].unique()
    unique_pull_labels_1 = label_1_df['Pull_Label'].unique()

    train_size_0 = int(0.8 * len(unique_pull_labels_0))
    train_size_1 = int(0.8 * len(unique_pull_labels_1))


    np.random.shuffle(unique_pull_labels_0)
    np.random.shuffle(unique_pull_labels_1)

    train_pull_labels_0 = unique_pull_labels_0[:train_size_0]
    test_pull_labels_0 = unique_pull_labels_0[train_size_0:]

    train_pull_labels_1 = unique_pull_labels_1[:train_size_1]
    test_pull_labels_1 = unique_pull_labels_1[train_size_1:]

    train_df_0 = label_0_df[label_0_df['Pull_Label'].isin(train_pull_labels_0)]
    test_df_0 = label_0_df[label_0_df['Pull_Label'].isin(test_pull_labels_0)]

    train_df_1 = label_1_df[label_1_df['Pull_Label'].isin(train_pull_labels_1)]
    test_df_1 = label_1_df[label_1_df['Pull_Label'].isin(test_pull_labels_1)]


    train_df = pd.concat([train_df_0, train_df_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat([test_df_0, test_df_1]).sample(frac=1, random_state=42).reset_index(drop=True)


    train_file_path = os.path.join(save_path, f"{dataset_name}_train.csv")
    test_file_path = os.path.join(save_path, f"{dataset_name}_test.csv")

    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    print(f"Train and test datasets have been successfully saved for '{dataset_name}'!")
    print(f"Train: {train_file_path}")
    print(f"Test: {test_file_path}")
