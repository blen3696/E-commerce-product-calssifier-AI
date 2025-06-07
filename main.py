import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('amazon-products.csv')
print("First 5 rows:\n", df.head())

# Check missing values in 'description' and 'categories'
print("\nMissing values:")
print(df[['description', 'categories']].isnull().sum())

# Drop rows with missing description or categories
df = df.dropna(subset=['description', 'categories']).reset_index(drop=True)

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning on description
df['clean_description'] = df['description'].apply(clean_text)

# Filter top categories (choose top 10 for example) as target
top_categories = df['categories'].value_counts().head(10).index
df = df[df['categories'].isin(top_categories)].reset_index(drop=True)

print("\nTop categories used:")
print(top_categories)

# Split data
X = df['clean_description']
y = df['categories']

# stratify ensures balanced category distribution in train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF Vectorization - to converts text to numerical vectors, capturing important words
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit on training data and transform both train and test sets
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model (with class_weight='balanced' to handle imbalance)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Macro F1-score: {f1_score(y_test, y_pred, average='macro'):.2f}")

# Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=top_categories)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=top_categories, yticklabels=top_categories, cmap="Blues")
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('Confusion Matrix - Product Categorization')
plt.show()


