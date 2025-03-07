import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import nltk
import os

# Download required NLTK data
nltk.download('punkt')

# Constants
VECTOR_SIZE = 100
HIDDEN_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Tokenization and Embedding
def random_token_embedding(tokens, vector_size):
    return [np.random.uniform(-1, 1, vector_size) for token in tokens]

def tokenize_and_embed(text):
    tokens = word_tokenize(text.lower())
    return random_token_embedding(tokens, VECTOR_SIZE)

# Dataset Preparation
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x1, x2, x3, y = self.data[idx]
        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(x3, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

# Custom collate function
def collate_fn(batch, max_len=50):
    x1, x2, x3, y = zip(*batch)
    x1 = nn.utils.rnn.pad_sequence([torch.tensor(seq[:max_len]) for seq in x1], batch_first=True)
    x2 = nn.utils.rnn.pad_sequence([torch.tensor(seq[:max_len]) for seq in x2], batch_first=True)
    x3 = nn.utils.rnn.pad_sequence([torch.tensor(seq[:max_len]) for seq in x3], batch_first=True)
    y = torch.tensor(y, dtype=torch.float32)
    return x1, x2, x3, y

# LSTM Model Definition
class DeepLinkModel(nn.Module):
    def __init__(self, hidden_size):
        super(DeepLinkModel, self).__init__()
        self.lstm1 = nn.LSTM(VECTOR_SIZE, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(VECTOR_SIZE, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(VECTOR_SIZE, hidden_size, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        _, (h1, _) = self.lstm1(x1)
        _, (h2, _) = self.lstm2(x2)
        _, (h3, _) = self.lstm3(x3)
        
        score1 = torch.cosine_similarity(h1[-1], h2[-1], dim=1)
        score2 = torch.cosine_similarity(h1[-1], h3[-1], dim=1)
        combined_score = torch.maximum(score1, score2).unsqueeze(1)
        output = self.sigmoid(combined_score)
        return output

# Training Function
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x1, x2, x3, y in dataloader:
        x1, x2, x3, y = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), y.unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x1, x2, x3)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation Function to Add Predictions
def predict(model, dataloader, df):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x1, x2, x3, _ in dataloader:
            x1, x2, x3 = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE)
            outputs = model(x1, x2, x3)
            predicted = (outputs.cpu().numpy() > 0.5).astype(int)
            predictions.extend(predicted.flatten())

    df['Predicted'] = predictions
    return df

# List of train and test dataset paths (paired)
dataset_paths = [
    # ('Add the train file of the dataset',
    #  'Add the test file of the dataset'),
]


# Process each dataset
for train_path, test_path in dataset_paths:
    try:
        dataset_name = os.path.basename(train_path).replace('_train.csv', '')
        print(f"\nProcessing dataset: {dataset_name}")

        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Prepare datasets
        train_sequences = [
            (tokenize_and_embed(row['Commit_Message']), 
             tokenize_and_embed(row['Pull_Request_Description']), 
             tokenize_and_embed(row['Pull_Request_Title'] + row['File_Changes']), 
             row['label']) 
            for _, row in train_df.iterrows()
        ]

        test_sequences = [
            (tokenize_and_embed(row['Commit_Message']), 
             tokenize_and_embed(row['Pull_Request_Description']), 
             tokenize_and_embed(row['Pull_Request_Title'] + row['File_Changes']), 
             row['label']) 
            for _, row in test_df.iterrows()
        ]

        train_loader = DataLoader(TextDataset(train_sequences), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(TextDataset(test_sequences), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        # Initialize Model
        model = DeepLinkModel(HIDDEN_SIZE).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training Loop
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss:.4f}")

        # Predict and Save Results
        test_df = predict(model, test_loader, test_df)
        output_path = f"/home/shubhi/commit/dataset/Python/results/deeplink/{dataset_name}_test_with_predictions.csv"
        test_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {train_path}: {e}")

print("Processing completed for all datasets.")




















