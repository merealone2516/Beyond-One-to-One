import pandas as pd
import torch
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Check GPU availability
def check_gpu_availability():
    try:
        if torch.cuda.is_available():
            print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU found. Using CPU instead.")
    except Exception as e:
        print(f"Error while checking GPU availability: {e}")

check_gpu_availability()



# List of train and test dataset paths (paired)
dataset_paths = [
    # ('Add the train file of the dataset',
    #  'Add the test file of the dataset'),
]


results_all_correct = []
results_at_least_one = []
results_at_least_two = []

for train_path, test_path in dataset_paths:
    try:
        dataset_name = os.path.basename(train_path).replace('_train.csv', '')
        print(f"\nProcessing {dataset_name}...")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_df['combined_text'] = train_df['Pull_Request_Title'] + " " + train_df['Pull_Request_Description'] + " " + train_df['Commit_Message'] + " " + train_df['File_Changes']
        test_df['combined_text'] = test_df['Pull_Request_Title'] + " " + test_df['Pull_Request_Description'] + " " + test_df['Commit_Message'] + " " + test_df['File_Changes']
        

        train_df['combined_text'] = train_df['combined_text'].apply(lambda x: " ".join(np.random.permutation(x.split())))
        test_df['combined_text'] = test_df['combined_text'].apply(lambda x: " ".join(np.random.permutation(x.split())))


        vectorizer = TfidfVectorizer(max_features=1000)
        X_train = vectorizer.fit_transform(train_df['combined_text'])
        y_train = train_df['label']
        X_test = vectorizer.transform(test_df['combined_text'])

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)


        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "logloss",
            "max_depth": 2, 
            "learning_rate": 0.001, 
            "colsample_bytree": 0.5,  
            "random_state": 42
        }
        model = xgb.train(params, dtrain, num_boost_round=10) 


        y_pred_prob = model.predict(dtest)
        test_df['Predicted'] = (y_pred_prob > 0.5).astype(int) 


        correct_counts = test_df.groupby('Pull_Label').apply(lambda x: (x['label'] == x['Predicted']).sum())


        all_correct = (correct_counts == test_df.groupby('Pull_Label').size()).sum() / len(correct_counts) * 100

        at_least_one_correct = (correct_counts >= 1).sum() / len(correct_counts) * 100


        at_least_two_correct = (correct_counts >= 2).sum() / len(correct_counts) * 100


        results_all_correct.append({"Dataset": dataset_name, "Accuracy": round(all_correct, 2)})
        results_at_least_one.append({"Dataset": dataset_name, "Accuracy": round(at_least_one_correct, 2)})
        results_at_least_two.append({"Dataset": dataset_name, "Accuracy": round(at_least_two_correct, 2)})

        print(f"Finished processing {dataset_name}\n")

    except Exception as e:
        print(f"Error processing {train_path} and {test_path}: {e}")

# Save results to CSV files
pd.DataFrame(results_all_correct).to_csv('path', index=False, float_format='%.2f')
pd.DataFrame(results_at_least_one).to_csv('path', index=False, float_format='%.2f')
pd.DataFrame(results_at_least_two).to_csv('path', index=False, float_format='%.2f')

print("Evaluation completed. Results saved in CSV files.")




















