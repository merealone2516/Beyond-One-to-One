import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(data):
    labels = data['label']
    predictions = data['Predicted']
    precision = round(precision_score(labels, predictions, zero_division=0) * 100, 2)
    recall = round(recall_score(labels, predictions, zero_division=0) * 100, 2)
    f1 = round(f1_score(labels, predictions, zero_division=0) * 100, 2)
    accuracy = round((labels == predictions).mean() * 100, 2)
    return accuracy, precision, recall, f1

def process_datasets(dataset_paths, output_path):

    all_correct_df = pd.DataFrame()
    at_least_one_df = pd.DataFrame()
    at_least_two_df = pd.DataFrame()


    for file_path in dataset_paths:

        data = pd.read_csv(file_path)


        total_unique_pull_labels = data['Pull_Label'].nunique()

        grouped = data.groupby('Pull_Label', group_keys=False)

        # Case 1: All values correctly predicted
        correct_all = grouped.apply(lambda group: (group['label'] == group['Predicted']).all()).sum()
        accuracy_all_correct = round((correct_all / total_unique_pull_labels) * 100, 2)

        # Case 2: At least one value correctly predicted
        correct_at_least_one = grouped.apply(lambda group: (group['label'] == group['Predicted']).any()).sum()
        accuracy_at_least_one = round((correct_at_least_one / total_unique_pull_labels) * 100, 2)

        # Case 3: At least two values correctly predicted
        correct_at_least_two = grouped.apply(lambda group: (group['label'] == group['Predicted']).sum() >= 2).sum()
        accuracy_at_least_two = round((correct_at_least_two / total_unique_pull_labels) * 100, 2)


        accuracy, precision, recall, f1 = calculate_metrics(data)


        all_correct_result = {
            'Dataset': file_path,
            'Accuracy': accuracy_all_correct,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        at_least_one_result = {
            'Dataset': file_path,
            'Accuracy': accuracy_at_least_one,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        at_least_two_result = {
            'Dataset': file_path,
            'Accuracy': accuracy_at_least_two,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }


        all_correct_df = pd.concat([all_correct_df, pd.DataFrame([all_correct_result])], ignore_index=True)
        at_least_one_df = pd.concat([at_least_one_df, pd.DataFrame([at_least_one_result])], ignore_index=True)
        at_least_two_df = pd.concat([at_least_two_df, pd.DataFrame([at_least_two_result])], ignore_index=True)


    # Save results to CSV files
    all_correct_df.to_csv(f"{output_path}/all_correct.csv", index=False)
    at_least_one_df.to_csv(f"{output_path}/at_least_one.csv", index=False)
    at_least_two_df.to_csv(f"{output_path}/at_least_two.csv", index=False)


# Define dataset paths
dataset_paths = [ Path of the datasets

]

output_path = 'Path to save results'
process_datasets(dataset_paths, output_path)