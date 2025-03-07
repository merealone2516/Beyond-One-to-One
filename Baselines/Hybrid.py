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























# import pandas as pd
# import torch
# import numpy as np
# import xgboost as xgb
# from sklearn.feature_extraction.text import TfidfVectorizer
# import os

# # Check GPU availability
# def check_gpu_availability():
#     try:
#         if torch.cuda.is_available():
#             print(f"GPU is available: {torch.cuda.get_device_name(0)}")
#         else:
#             print("No GPU found. Using CPU instead.")
#     except Exception as e:
#         print(f"Error while checking GPU availability: {e}")

# check_gpu_availability()


# dataset_paths = [
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_activiti_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_activiti_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_alibaba_nacos_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_alibaba_nacos_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_Anuken-mindustry_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_Anuken-mindustry_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_dolphinscheduler_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_dolphinscheduler_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_dubbo_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_dubbo_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_fineract_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_fineract_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_flink_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_flink_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_incubator_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_incubator_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_kafka_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_kafka_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_pinot_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_pinot_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_rocketmq_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_rocketmq_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_shardingsphere_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_shardingsphere_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_skywalking_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_apache_skywalking_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_bazelbuild_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_bazelbuild_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_dbeaver_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_dbeaver_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_debezium_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_debezium_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_dotcms_core_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_dotcms_core_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_elastic_logstash_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_elastic_logstash_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_elastic_search_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_elastic_search_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_elastic_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_elastic_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_eugenp_tutorials_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_eugenp_tutorials_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_halo-dev_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_halo-dev_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_jabref_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_jabref_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_jenkinsci_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_jenkinsci_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_jetbrains_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_jetbrains_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_keycloak_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_keycloak_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_libgdx_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_libgdx_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_neo4j_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_neo4j_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_open-telemetry_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_open-telemetry_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_openapitools_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_openapitools_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_openjdk_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_openjdk_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_openrefine_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_openrefine_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_opensearch_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_opensearch_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_oracle-graal_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_oracle-graal_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_selenium_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_selenium_test.csv'),
#     ('/home/shubhi/commit/dataset/Java/new/Updated_Final_spring-boot_train.csv',
#      '/home/shubhi/commit/dataset/Java/new/Updated_Final_spring-boot_test.csv')
# ]

# # List of train and test dataset paths (paired)
# # # List of train and test dataset paths (paired)
# # dataset_paths = [
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_ansible_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_ansible_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_commai_openpilot_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_commai_openpilot_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_cpython_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_cpython_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_django_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_django_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_fastapi_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_fastapi_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_getsentry_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_getsentry_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_home-assistant-core_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_home-assistant-core_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_huggingface-transformers_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_huggingface-transformers_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_keras_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_keras_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_localstack_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_localstack_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_matplotlib_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_matplotlib_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_numpy_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_numpy_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_Pandas_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_Pandas_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_pytorch_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_pytorch_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_scikitlearn_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_scikitlearn_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_scrapy_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_scrapy_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_significant-gravitas-autogpt_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_significant-gravitas-autogpt_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_tensorflow_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_tensorflow_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_TheAlgorithms_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_TheAlgorithms_test.csv'),
# #     ('/home/shubhi/commit/dataset/Python/new/Updated_Final_yt_dlp_train.csv',
# #      '/home/shubhi/commit/dataset/Python/new/Updated_Final_yt_dlp_test.csv')
# # ]



# # Function to calculate precision, recall, f1, and accuracy for a given condition
# def calculate_metrics(condition):
#     total_pull_labels = len(condition)
#     true_positives = condition.sum()
#     true_negatives = (~condition).sum()
    
#     # accuracy = true_positives / total_pull_labels
#     # precision = true_positives / (true_positives + true_negatives) if true_positives + true_negatives > 0 else 0
#     # recall = true_positives / total_pull_labels
#     # f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    
#     accuracy = round((true_positives / total_pull_labels) * 100, 2)
#     precision = round((true_positives / (true_positives + true_negatives) * 100) if true_positives + true_negatives > 0 else 0, 2)
#     recall = round((true_positives / total_pull_labels) * 100, 2)
#     f1 = round((2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0, 2)

#     return {
#         "Accuracy": accuracy,
#         "Precision": precision,
#         "Recall": recall,
#         "F1-Score": f1
#     }

# # Processing datasets
# results_all_correct = []
# results_at_least_one = []
# results_at_least_two = []
# results_none_correct = []

# for train_path, test_path in dataset_paths:
#     try:
#         dataset_name = os.path.basename(train_path).replace('_train.csv', '')
#         print(f"\nProcessing {dataset_name}...")

#         # Load datasets
#         train_df = pd.read_csv(train_path)
#         test_df = pd.read_csv(test_path)

#         # Combine text columns into a single feature
#         train_df['combined_text'] = train_df['Pull_Request_Title'] + " " + train_df['Pull_Request_Description'] + " " + train_df['Commit_Message'] + " " + train_df['File_Changes']
#         test_df['combined_text'] = test_df['Pull_Request_Title'] + " " + test_df['Pull_Request_Description'] + " " + test_df['Commit_Message'] + " " + test_df['File_Changes']

#         # TF-IDF Vectorization
#         vectorizer = TfidfVectorizer(max_features=5000)
#         X_train = vectorizer.fit_transform(train_df['combined_text'])
#         y_train = train_df['label']
#         X_test = vectorizer.transform(test_df['combined_text'])

#         # Convert data to DMatrix for XGBoost
#         dtrain = xgb.DMatrix(X_train, label=y_train)
#         dtest = xgb.DMatrix(X_test)

#         # Train the model using GPU
#         params = {
#             "objective": "binary:logistic",
#             "tree_method": "hist",
#             "eval_metric": "logloss",
#             "random_state": 42
#         }
#         model = xgb.train(params, dtrain, num_boost_round=100)

#         # Predict on the test set
#         y_pred_prob = model.predict(dtest)
#         test_df['Predicted'] = (y_pred_prob > 0.5).astype(int)

#         # Save predictions for further analysis
#         # test_df.to_csv(f'/home/shubhi/commit/dataset/Javascript/results/hybrid/{dataset_name}_predictions.csv', index=False)

#         # 1. All values correctly predicted
#         all_correct = test_df.groupby('Pull_Label').apply(
#             lambda x: (x['label'] == x['Predicted']).all()
#         )
#         # 2. At least one value correctly predicted
#         at_least_one_correct = test_df.groupby('Pull_Label').apply(
#             lambda x: (x['label'] == x['Predicted']).any()
#         )
#         # 3. At least two values correctly predicted
#         at_least_two_correct = test_df.groupby('Pull_Label').apply(
#             lambda x: (x['label'] == x['Predicted']).sum() >= 2
#         )
#         # 4. None of the values correctly predicted
#         none_correct = test_df.groupby('Pull_Label').apply(
#             lambda x: (x['label'] == x['Predicted']).sum() == 0
#         )

#         # Calculate metrics for each condition
#         metrics_all_correct = calculate_metrics(all_correct)
#         metrics_at_least_one_correct = calculate_metrics(at_least_one_correct)
#         metrics_at_least_two_correct = calculate_metrics(at_least_two_correct)
#         metrics_none_correct = calculate_metrics(none_correct)

#         # Append results
#         results_all_correct.append({"Dataset": dataset_name, **metrics_all_correct})
#         results_at_least_one.append({"Dataset": dataset_name, **metrics_at_least_one_correct})
#         results_at_least_two.append({"Dataset": dataset_name, **metrics_at_least_two_correct})
#         results_none_correct.append({"Dataset": dataset_name, **metrics_none_correct})

#         print(f"Finished processing {dataset_name}\n")

#     except Exception as e:
#         print(f"Error processing {train_path} and {test_path}: {e}")

# # Save results to CSV files
# pd.DataFrame(results_all_correct).to_csv('/home/shubhi/commit/dataset/Java/results/re_hybrid/results_all_correct.csv', index=False)
# pd.DataFrame(results_at_least_one).to_csv('/home/shubhi/commit/dataset/Java/results/re_hybrid/results_at_least_one.csv', index=False)
# pd.DataFrame(results_at_least_two).to_csv('/home/shubhi/commit/dataset/Java/results/re_hybrid/results_at_least_two.csv', index=False)
# pd.DataFrame(results_none_correct).to_csv('/home/shubhi/commit/dataset/Java/results/re_hybrid/results_none_correct.csv', index=False)

# print("Evaluation completed. Results saved in CSV files.")















# import pandas as pd
# import torch
# import numpy as np
# import xgboost as xgb
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import StratifiedKFold
# import os

# # Check GPU availability
# def check_gpu_availability():
#     try:
#         if torch.cuda.is_available():
#             print(f"GPU is available: {torch.cuda.get_device_name(0)}")
#         else:
#             print("No GPU found. Using CPU instead.")
#     except Exception as e:
#         print(f"Error while checking GPU availability: {e}")

# check_gpu_availability()

# # # Dataset paths
# dataset_paths = [
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_Apache_mxnet_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_Apache_mxnet_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_apollo-auto_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_apollo-auto_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_Ardupilot_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_Ardupilot_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_Bitcoin_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_Bitcoin_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_Clickhouse_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_Clickhouse_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_dmlc_xgboost_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_dmlc_xgboost_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_duckdb_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_duckdb_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_electron_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_electron_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_emscripten_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_emscripten_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_ethereum_solidity_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_ethereum_solidity_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_facebook-react-native_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_facebook-react-native_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_facebook-rocksdb_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_facebook-rocksdb_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_FreeCAD_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_FreeCAD_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_ggerganov_llama_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_ggerganov_llama_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_godotengine_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_godotengine_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_grpc_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_grpc_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_microsoft_terminal_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_microsoft_terminal_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_microsoft_winget_cli_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_microsoft_winget_cli_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_opencv_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_opencv_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_openvinotoolkit_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_openvinotoolkit_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_osquery_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_osquery_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_Paddle_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_Paddle_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_projectchip_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_projectchip_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_protocolbuffers_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_protocolbuffers_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_qgis_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_qgis_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_swiftlang_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_swiftlang_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_taichi-dev_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_taichi-dev_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_tensorflow_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_tensorflow_test.csv'),
#     ('/home/shubhi/commit/dataset/C++/new/Updated_Final_xbmc_train.csv',
#      '/home/shubhi/commit/dataset/C++/new/Updated_Final_xbmc_test.csv')
# ]

# # Custom function for group metrics evaluation
# def evaluate_group_metrics(results_df):
#     grouped = results_df.groupby('Pull_Label')
#     precision_scores = []
#     recall_scores = []
#     f1_scores = []
#     percentage_correct = []

#     for name, group in grouped:
#         true_links = group[group['Actual'] == 1].index.to_numpy()
#         predicted_links = group[group['Predicted'] == 1].index.to_numpy()

#         tp = len(set(predicted_links) & set(true_links))
#         fp = len(set(predicted_links) - set(true_links))
#         fn = len(set(true_links) - set(predicted_links))

#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

#         percentage = (tp / len(true_links)) * 100 if len(true_links) > 0 else 0

#         precision_scores.append(precision)
#         recall_scores.append(recall)
#         f1_scores.append(f1)
#         percentage_correct.append(percentage)

#     avg_precision = np.mean(precision_scores)
#     avg_recall = np.mean(recall_scores)
#     avg_f1 = np.mean(f1_scores)
#     avg_percentage_correct = np.mean(percentage_correct)

#     return avg_precision, avg_recall, avg_f1, avg_percentage_correct

# # Initialize results storage
# results = []

# print("Processing datasets...\n")

# # Loop through datasets
# for dataset_path in dataset_paths:
#     try:
#         print(f"Processing {os.path.basename(dataset_path)}...")

#         # Load dataset
#         df = pd.read_csv(dataset_path)

#         # Combine text columns into a single feature
#         df['combined_text'] = (
#             df['Pull_Request_Title'] + ' ' +
#             df['Pull_Request_Description'] + ' ' +
#             df['Commit_Message'] + ' ' +
#             df['File_Changes']
#         )

#         # Preprocess text using TF-IDF vectorization
#         vectorizer = TfidfVectorizer(max_features=5000)
#         X = vectorizer.fit_transform(df['combined_text'])
#         y = df['label']
#         pull_labels = df['Pull_Label']

#         # Initialize 5-Fold Cross-Validation
#         skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         fold_metrics = []

#         # Cross-Validation Loop
#         for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
#             # Split data
#             X_train, X_test = X[train_idx], X[test_idx]
#             y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#             # Convert data to DMatrix for XGBoost
#             dtrain = xgb.DMatrix(X_train, label=y_train)
#             dtest = xgb.DMatrix(X_test, label=y_test)

#             # Train Gradient Boosting Model
#             params = {
#                 "objective": "binary:logistic",
#                 "tree_method": "hist",
#                 "eval_metric": "logloss",
#                 "random_state": 42
#             }
#             model = xgb.train(params, dtrain, num_boost_round=100)

#             # Predict probabilities
#             y_pred_prob = model.predict(dtest)
#             y_pred = (y_pred_prob > 0.5).astype(int)

#             # Evaluate Group-Level Performance
#             results_df = df.iloc[test_idx].copy()
#             results_df['Actual'] = y_test.values
#             results_df['Predicted'] = y_pred

#             avg_precision, avg_recall, avg_f1, avg_percentage_correct = evaluate_group_metrics(results_df)
#             fold_metrics.append({
#                 'precision': avg_precision,
#                 'recall': avg_recall,
#                 'f1': avg_f1,
#                 'percentage_correct': avg_percentage_correct
#             })

#         # Average Metrics Across Folds
#         avg_metrics = {k: np.mean([fold[k] for fold in fold_metrics]) for k in fold_metrics[0]}
#         results.append({
#             'Dataset': os.path.basename(dataset_path),
#             **avg_metrics
#         })

#         print(f"Completed {os.path.basename(dataset_path)}\n")

#     except Exception as e:
#         print(f"Error processing {dataset_path}: {e}\n")

# # Save results to CSV
# results_df = pd.DataFrame(results)
# results_df.to_csv('/home/shubhi/commit/dataset/C++/results_summary_HybridLink_C++.csv', index=False)
# print("Results saved to evaluation_results.csv")





















# import pandas as pd
# import torch
# import numpy as np
# import xgboost as xgb
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import StratifiedKFold

# # Check GPU availability
# def check_gpu_availability():
#     try:
#         if torch.cuda.is_available():
#             print(f"GPU is available: {torch.cuda.get_device_name(0)}")
#         else:
#             print("No GPU found. Using CPU instead.")
#     except Exception as e:
#         print(f"Error while checking GPU availability: {e}")


# check_gpu_availability()

# # Load dataset
# file_path = '/home/shubhi/commit/dataset/Updated_Final_home-assistant-core.csv'  # Update with your dataset path
# df = pd.read_csv(file_path)

# # Combine text columns into a single feature
# df['combined_text'] = (
#     df['Pull_Request_Title'] + ' ' +
#     df['Pull_Request_Description'] + ' ' +
#     df['Commit_Message'] + ' ' +
#     df['File_Changes']
# )

# # Preprocess text using TF-IDF vectorization
# vectorizer = TfidfVectorizer(max_features=5000)
# X = vectorizer.fit_transform(df['combined_text'])
# y = df['label']
# pull_labels = df['Pull_Label']

# # Custom function for group metrics evaluation
# def evaluate_group_metrics(results_df):
#     grouped = results_df.groupby('Pull_Label')
#     precision_scores = []
#     recall_scores = []
#     f1_scores = []
#     percentage_correct = []

#     for name, group in grouped:
#         true_links = group[group['Actual'] == 1].index.to_numpy()
#         predicted_links = group[group['Predicted'] == 1].index.to_numpy()

#         tp = len(set(predicted_links) & set(true_links))
#         fp = len(set(predicted_links) - set(true_links))
#         fn = len(set(true_links) - set(predicted_links))

#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

#         percentage = (tp / len(true_links)) * 100 if len(true_links) > 0 else 0

#         precision_scores.append(precision)
#         recall_scores.append(recall)
#         f1_scores.append(f1)
#         percentage_correct.append(percentage)

#     avg_precision = np.mean(precision_scores)
#     avg_recall = np.mean(recall_scores)
#     avg_f1 = np.mean(f1_scores)
#     avg_percentage_correct = np.mean(percentage_correct)

#     return avg_precision, avg_recall, avg_f1, avg_percentage_correct

# # Initialize 5-Fold Cross-Validation
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# fold_metrics = []

# print("Starting 5-Fold Cross-Validation...\n")

# # Cross-Validation Loop
# for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
#     print(f"Fold {fold}")

#     # Split data
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#     # Convert data to DMatrix for XGBoost
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dtest = xgb.DMatrix(X_test, label=y_test)

#     # Train Gradient Boosting Model
#     print("Training Gradient Boosting Model on GPU...")
#     params = {
#         "objective": "binary:logistic",
#         "tree_method": "hist",  # GPU-based optimization
#         "eval_metric": "logloss",
#         "random_state": 42
#     }
#     model = xgb.train(params, dtrain, num_boost_round=100)

#     # Predict probabilities
#     y_pred_prob = model.predict(dtest)
#     y_pred = (y_pred_prob > 0.5).astype(int)

#     # Evaluate Group-Level Performance
#     results_df = df.iloc[test_idx].copy()
#     results_df['Actual'] = y_test.values
#     results_df['Predicted'] = y_pred

#     avg_precision, avg_recall, avg_f1, avg_percentage_correct = evaluate_group_metrics(results_df)

#     print(f"Fold {fold}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1-Score={avg_f1:.4f}, Percentage Correct={avg_percentage_correct:.2f}%\n")
#     fold_metrics.append({'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1, 'percentage_correct': avg_percentage_correct})

# # Average Metrics Across Folds
# avg_metrics = {k: np.mean([fold[k] for fold in fold_metrics]) for k in fold_metrics[0]}
# print("\nAverage Metrics Across Folds:")
# for metric, value in avg_metrics.items():
#     print(f"{metric.capitalize()}: {value:.4f}")
