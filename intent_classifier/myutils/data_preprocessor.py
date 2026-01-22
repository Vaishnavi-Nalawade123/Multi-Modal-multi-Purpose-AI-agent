import os
import pandas as pd
import string
import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def load_and_prepare_data(folder_path):
    all_data = []

    for file in glob.glob(os.path.join(folder_path, "*.csv")):
        df = pd.read_csv(file)
        df = df[['text', 'intent']].dropna()
        df['text'] = df['text'].apply(clean_text)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df
