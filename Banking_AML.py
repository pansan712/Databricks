import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate dummy data
np.random.seed(42)

def generate_dummy_data(n_samples=10000):
    data = {
        'transaction_id': range(1, n_samples + 1),
        'amount': np.random.exponential(1000, n_samples),
        'sender_account': np.random.randint(10000, 99999, n_samples),
        'receiver_account': np.random.randint(10000, 99999, n_samples),
        'transaction_type': np.random.choice(['transfer', 'deposit', 'withdrawal'], n_samples),
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    }
    
    df = pd.DataFrame(data)
    
    # Generate some anomalies (5% of the data)
    anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    df.loc[anomaly_indices, 'amount'] *= np.random.uniform(10, 100, len(anomaly_indices))
    
    # Add a label column (1 for anomaly, 0 for normal)
    df['label'] = 0
    df.loc[anomaly_indices, 'label'] = 1
    
    return df

# Generate and display dummy data
df = generate_dummy_data()
print(df.head())
print(f"\nShape of the dataset: {df.shape}")
print(f"\nDistribution of labels:\n{df['label'].value_counts(normalize=True)}")

# Prepare features for modeling
features = ['amount']
X = df[features]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Isolation Forest model
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X_train_scaled)

# Predict anomalies
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)

# Convert predictions to binary (1 for anomaly, 0 for normal)
y_pred_train = np.where(y_pred_train == -1, 1, 0)
y_pred_test = np.where(y_pred_test == -1, 1, 0)

# Evaluate the model
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test))

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred_test))

# Visualize the results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='timestamp', y='amount', hue='label', alpha=0.6)
plt.title('Transaction Amount over Time')
plt.xlabel('Timestamp')
plt.ylabel('Amount')
plt.yscale('log')
plt.show()

# Visualize the anomaly scores
anomaly_scores = -clf.score_samples(X_test_scaled)
plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores, bins=50)
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.show()

# Function to predict if a new transaction is anomalous
def predict_anomaly(transaction_amount):
    scaled_amount = scaler.transform([[transaction_amount]])
    prediction = clf.predict(scaled_amount)
    return "Anomalous" if prediction == -1 else "Normal"

# Example usage
new_transaction_amount = 50000
result = predict_anomaly(new_transaction_amount)
print(f"\nPrediction for transaction amount ${new_transaction_amount}: {result}")
