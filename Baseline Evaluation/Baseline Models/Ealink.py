import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import RobertaModel, RobertaTokenizer
import os
from torch.cuda.amp import GradScaler, autocast

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
max_seq_len = 128

# Batch tokenize texts
def batch_tokenize_texts(texts, max_length):
    encoded = tokenizer(list(texts), padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    return encoded['input_ids'], encoded['attention_mask']

# Custom Dataset class
class IssueCommitDataset(Dataset):
    def __init__(self, issue_tokens, commit_tokens, labels=None):
        self.issue_tokens = issue_tokens
        self.commit_tokens = commit_tokens
        self.labels = labels

    def __len__(self):
        return len(self.issue_tokens)

    def __getitem__(self, idx):
        issue_input = self.issue_tokens[idx]
        commit_input = self.commit_tokens[idx]
        label = self.labels[idx] if self.labels is not None else None
        return issue_input[0], issue_input[1], commit_input[0], commit_input[1], label

# Model Definition with Batch Normalization and No Sigmoid in Forward
class EALink(nn.Module):
    def __init__(self):
        super(EALink, self).__init__()
        self.codebert = RobertaModel.from_pretrained('microsoft/codebert-base')
        self.fc1 = nn.Linear(768 * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, issue_input, issue_mask, commit_input, commit_mask):
        issue_out = self.codebert(input_ids=issue_input, attention_mask=issue_mask).pooler_output
        commit_out = self.codebert(input_ids=commit_input, attention_mask=commit_mask).pooler_output

        combined = torch.cat((issue_out, commit_out), dim=1)
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.dropout(torch.relu(x))
        x = self.fc2(x)
        return x  # No sigmoid here, handled in the loss function

# Training Function with Mixed Precision
def train_model(model, train_loader, criterion, optimizer, epochs, accumulation_steps):
    model.train()
    scaler = GradScaler()
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch [{epoch+1}/{epochs}]")
            optimizer.zero_grad()

            for i, (issue_input, issue_mask, commit_input, commit_mask, labels) in enumerate(tepoch):
                issue_input, issue_mask = issue_input.to(device), issue_mask.to(device)
                commit_input, commit_mask = commit_input.to(device), commit_mask.to(device)
                labels = labels.to(device).float().view(-1, 1)

                with autocast():
                    outputs = model(issue_input, issue_mask, commit_input, commit_mask)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item()
                tepoch.set_postfix(loss=(running_loss / (tepoch.n + 1)))

# Prediction Function
def predict(model, dataloader, test_df):
    model.eval()
    predictions = []
    with torch.no_grad():
        for issue_input, issue_mask, commit_input, commit_mask, _ in dataloader:
            issue_input, issue_mask = issue_input.to(device), issue_mask.to(device)
            commit_input, commit_mask = commit_input.to(device), commit_mask.to(device)

            outputs = model(issue_input, issue_mask, commit_input, commit_mask)
            predicted_probs = torch.sigmoid(outputs).cpu().numpy()
            predicted = (predicted_probs > 0.5).astype(int).flatten()
            predictions.extend(predicted)

    test_df['Predicted'] = predictions
    return test_df

# Dataset paths paired
dataset_paths = [
    # ('Add the train file of the dataset',
    #  'Add the test file of the dataset'),
]

# Process each dataset
for train_path, test_path in dataset_paths:
    try:
        dataset_name = os.path.basename(train_path).replace('_train.csv', '')
        print(f"\nProcessing dataset: {dataset_name}")


        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_issue_input, train_issue_mask = batch_tokenize_texts(train_df['Pull_Request_Title'] + ' ' + train_df['Pull_Request_Description'], max_seq_len)
        train_commit_input, train_commit_mask = batch_tokenize_texts(train_df['Commit_Message'] + ' ' + train_df['File_Changes'], max_seq_len)

        test_issue_input, test_issue_mask = batch_tokenize_texts(test_df['Pull_Request_Title'] + ' ' + test_df['Pull_Request_Description'], max_seq_len)
        test_commit_input, test_commit_mask = batch_tokenize_texts(test_df['Commit_Message'] + ' ' + test_df['File_Changes'], max_seq_len)

        train_dataset = IssueCommitDataset(list(zip(train_issue_input, train_issue_mask)), list(zip(train_commit_input, train_commit_mask)), train_df['label'].values)
        test_dataset = IssueCommitDataset(list(zip(test_issue_input, test_issue_mask)), list(zip(test_commit_input, test_commit_mask)), test_df['label'].values)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


        model = EALink().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

        train_model(model, train_loader, criterion, optimizer, epochs=5, accumulation_steps=4)


        test_df_with_predictions = predict(model, test_loader, test_df)

        output_path = f"/home/shubhi/commit/dataset/C++/Ealink/{dataset_name}_predictions.csv"
        test_df_with_predictions.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {train_path}: {e}")

print("Processing completed for all datasets.")















