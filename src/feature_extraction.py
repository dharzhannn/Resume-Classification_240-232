
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# File paths
INPUT_PATH = "outputs/results/resume_extracted.csv"
OUTPUT_PATH = "outputs/results/tfidf_data.pkl"

print("=" * 50)
print("       TF-IDF FEATURE EXTRACTION")
print("=" * 50)

# Load extracted resume dataset
df = pd.read_csv(INPUT_PATH)

# Check missing values
df = df.dropna(subset=["resume_text", "category"])

# Remove duplicate records
df = df.drop_duplicates(
    subset=["resume_text", "category"]
)

# Separate text and labels
X = df["resume_text"]
y = df["category"]

print("\nTotal resumes:", len(df))
print("Total categories:", y.nunique())

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

# Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF EXTRACTION COMPLETED")
print("-" * 40)

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)

print("Vocabulary size:", len(tfidf.vocabulary_))

# Save all features and labels
os.makedirs("outputs/results", exist_ok=True)

joblib.dump({
    "X_train": X_train_tfidf,
    "X_test": X_test_tfidf,
    "y_train": y_train,
    "y_test": y_test,
    "vectorizer": tfidf
}, OUTPUT_PATH)

print("\nSaved TF-IDF data to:", OUTPUT_PATH)
