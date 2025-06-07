# Import necessary libraries
import pandas as pd                
import numpy as np                  
import re                           
import string                       

# Scikit-learn modules for preprocessing, model training, and evaluation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from a CSV file
df = pd.read_csv('amazon-products.csv')

# Display the first 5 rows to understand the structure of the dataset
print("First 5 rows:\n", df.head())

# Check for missing values specifically in the columns we're interested in: 'description' and 'categories'
print("\nMissing values:")
print(df[['description', 'categories']].isnull().sum())

# Remove rows that have missing descriptions or categories since they're not useful for classification
df = df.dropna(subset=['description', 'categories']).reset_index(drop=True)

# Function to clean the product description text
def clean_text(text):
    text = str(text).lower()  # Convert all text to lowercase for uniformity
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace and strip leading/trailing spaces
    return text

# Apply the text cleaning function to the 'description' column and create a new column 'clean_description'
df['clean_description'] = df['description'].apply(clean_text)

# Identify the top 10 most frequent product categories for model training as our target
top_categories = df['categories'].value_counts().head(10).index

# Filter the dataset to include only the products that fall under the top 10 categories
df = df[df['categories'].isin(top_categories)].reset_index(drop=True)

# Print the selected top categories
print("\nTop categories used:")
print(top_categories)

# Define feature (X) and label (y)
X = df['clean_description']  # Input features - cleaned product descriptions
y = df['categories']         # Target labels - product categories

# Split the data into training and test sets (80% train, 20% test)
# Use stratify to ensure class distribution is balanced in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Use TF-IDF vectorization to convert text to numerical feature vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit the vectorizer on training data and transform both train and test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train a Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Use the trained model to predict the categories of the test set
y_pred = model.predict(X_test_tfidf)

# Print a detailed classification report: precision, recall, F1-score for each class
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Print overall accuracy and macro-averaged F1 score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Macro F1-score: {f1_score(y_test, y_pred, average='macro'):.2f}")

# Create and visualize the confusion matrix to evaluate classification performance
cm = confusion_matrix(y_test, y_pred, labels=top_categories)
plt.figure(figsize=(10, 7))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    xticklabels=top_categories, 
    yticklabels=top_categories, 
    cmap="Blues"
)
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('Confusion Matrix - Product Categorization')
plt.show()


