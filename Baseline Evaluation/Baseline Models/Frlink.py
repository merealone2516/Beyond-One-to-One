import pandas as pd
import cupy as cp
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Check if GPU is available
def check_gpu_availability():
    try:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        if num_gpus > 0:
            print(f"GPU is available. Number of GPUs: {num_gpus}")
        else:
            print("No GPU found. Using CPU instead.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        print("CUDA runtime error. Using CPU instead.")
        print(e)

check_gpu_availability()


def get_similarity_batch(test_matrix, train_matrix, batch_size=1000):
    num_test_samples = test_matrix.shape[0]
    max_similarities = []
    for start in range(0, num_test_samples, batch_size):
        end = min(start + batch_size, num_test_samples)
        batch = test_matrix[start:end]
        dot_product = batch.dot(train_matrix.T)
        max_similarity = dot_product.max(axis=1).toarray().flatten()
        max_similarities.extend(max_similarity)
    return cp.asarray(max_similarities)


def calculate_metrics(labels, predictions):
    TP = ((labels == 1) & (predictions == 1)).sum()
    FP = ((labels == 0) & (predictions == 1)).sum()
    FN = ((labels == 1) & (predictions == 0)).sum()
    TN = ((labels == 0) & (predictions == 0)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

# Initialize storage
results_all_correct = []
results_at_least_one = []
results_at_least_two = []
results_none_correct = []

# List of train and test dataset paths (paired)
dataset_paths = [
    # ('Add the train file of the dataset',
    #  'Add the test file of the dataset'),
]

# Loop through datasets
for train_path, test_path in dataset_paths:
    dataset_name = os.path.basename(train_path).replace('_train.csv', '')
    print(f"Processing dataset: {dataset_name}")


    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)


    train_df['combined_text'] = train_df['Pull_Request_Title'] + " " + train_df['Pull_Request_Description'] + " " + train_df['Commit_Message'] + " " + train_df['File_Changes']
    test_df['combined_text'] = test_df['Pull_Request_Title'] + " " + test_df['Pull_Request_Description'] + " " + test_df['Commit_Message'] + " " + test_df['File_Changes']

    X_train = train_df['combined_text']
    y_train = cp.asarray(train_df['label'])
    X_test = test_df['combined_text']


    tfidf = TfidfVectorizer(max_features=1000)
    train_corpus = tfidf.fit_transform(X_train)
    test_corpus = tfidf.transform(X_test)


    train_corpus_gpu = cp.sparse.csr_matrix(train_corpus)
    test_corpus_gpu = cp.sparse.csr_matrix(test_corpus)

    try:
        similarities = get_similarity_batch(test_corpus_gpu, train_corpus_gpu, batch_size=1000)
        test_df['Predicted'] = [1 if sim >= 0.5 else 0 for sim in similarities.get()]
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"CUDA Runtime Error: {e}")
        continue

    # Save test set with predictions
    output_path = f'/home/shubhi/commit/dataset/C++/results_Frlink/{dataset_name}_test_with_predictions.csv'
    test_df.to_csv(output_path, index=False)



print("Processing completed for all datasets.")
























