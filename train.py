"""
train.py : Train XGBoost model on penguins dataset by sns and save it.

This script loads the Seaborn penguins dataset, preprocesses it using one-hot encoding
for input features and label encoding for the target variable, trains an XGBoost classifier
with parameters to prevent overfitting, evaluates it using F1-score, and saves the trained
model along with metadata to 'app/data/model.json'.

"""

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
import joblib
import os

""" Load penguin Dataset """

# code to load dataset
df = sns.load_dataset("penguins").dropna()
print(df.head())

# Initial shape
print(f"Initial dataset shape: {df.shape}")

""" Clean dataset by dropping missing values and duplicates """

# Drop rows with missing values
df.dropna(inplace=True)
print(f"After dropping missing values: {df.shape}")

# Drop duplicates if any
df.drop_duplicates(inplace=True)
print(f"After dropping duplicates: {df.shape}")

# Check data types of all columns
print("\nColumn Data Types:")
print(df.dtypes)

# Check unique values in 'island' and 'sex'
print("\nUnique values in 'island':", df['island'].unique())
print("\nUnique values in 'sex':", df['sex'].unique())
print("\nUnique values in 'species':", df['species'].unique())


"""
Preprocess dataset using one-hot encoding for input features
and label encoding for the target variable.

"""

# One-hot encode input categorical features: 'sex' and 'island'
df = pd.get_dummies(df, columns=["sex", "island"], drop_first=False)

# Label encode target variable: 'species'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["species_label"] = le.fit_transform(df["species"])

# Optional: see how classes were encoded
print("Species classes:", le.classes_)

"""
splits the data into train and test sets (80/20), trains an XGBoost model,
and saves the trained model with metadata.

"""

# Define features and target
X = df.drop(columns=["species", "species_label"])
y = df["species_label"]

# Stratified train-test split to preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

def train_and_evaluate(X_train, y_train, X_test, y_test, label_encoder):
    """
    Trains and evaluates XGBoost model, prints F1 scores and saves it to disk.
    """
    print("Starting model training...")

    model = XGBClassifier(
        max_depth=2,
        n_estimators=10,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    print("Model training complete.")

    # Predict on training and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Compute weighted F1-score
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')

    # Print scores
    print("\\n--- Evaluation Metrics ---")
    print(f"Train F1 Score: {f1_train:.4f}")
    print(f"Test F1 Score:  {f1_test:.4f}")
    print("\\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

    # Save model and metadata
    os.makedirs("app/data", exist_ok=True)
    joblib.dump({
        "model": model,
        "label_encoder": label_encoder,
        "columns": X_train.columns.tolist()
    }, "app/data/model.json")

    #print("\n Model trained and saved")
    print("\n Model saved to app/data/model.json")

# Call the function
train_and_evaluate(X_train, y_train, X_test, y_test, le)