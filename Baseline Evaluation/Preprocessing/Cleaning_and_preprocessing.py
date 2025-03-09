import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (if you haven't already)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess text
def preprocess_text(text):

    if pd.isnull(text):
        return ""

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)


def preprocess_dataset(file_path, save_path):

    df = pd.read_csv(file_path)

    df['Pull_Request_Title'] = df['Pull_Request_Title'].apply(preprocess_text)
    df['Pull_Request_Description'] = df['Pull_Request_Description'].apply(preprocess_text)
    df['Commit_Message'] = df['Commit_Message'].apply(preprocess_text)

    df.to_csv(save_path, index=False)


    print(f"First few rows of the preprocessed data from {file_path}:")
    print(df[['Pull_Request_Title', 'Pull_Request_Description', 'Commit_Message']].head())

# Preprocess both true and false datasets
preprocess_dataset('The path of the true files obtained after running the Python script Splitting_True_False.py', 'path to save the files obtained')
preprocess_dataset('The path of the false files obtained after running the Python script Splitting_True_False.py', 'path to save the files obtained')
