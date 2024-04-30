from flask import Flask, render_template
import numpy as np
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

app = Flask(__name__)
CORS(app)

df = pd.read_csv('msft_daily_prices.csv')

features = ['1. open', '4. close', '2. high', '3. low']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split the data into train and test sets
X_train, X_test = train_test_split(df[features], test_size=0.3, random_state=42)

# Train the One-Class SVM model on the training data
model_svm = OneClassSVM(nu=0.05)
model_svm.fit(X_train)

# Calculate anomaly scores for the test data using One-Class SVM
anomaly_scores_svm_train = model_svm.decision_function(X_train)
anomaly_scores_svm_test = model_svm.decision_function(X_test)

# Calculate the threshold using the training data anomaly scores
threshold_train = np.percentile(anomaly_scores_svm_train, 5)  # Example: consider 5th percentile as threshold
threshold_test = np.percentile(anomaly_scores_svm_test, 5)

# Flag anomalies based on the threshold
is_anomaly_train = anomaly_scores_svm_train < threshold_train
is_anomaly_test = anomaly_scores_svm_test < threshold_test
is_anomaly_svm = np.concatenate((is_anomaly_train, is_anomaly_test))

df['is_anomaly_svm'] = is_anomaly_svm

# Fit the Isolation Forest model
clf_if = IsolationForest(contamination=0.1, random_state=42)
clf_if.fit(X_train)

# Calculate anomaly scores for the test data using Isolation Forest
anomaly_scores_if_train = clf_if.decision_function(X_train)
anomaly_scores_if_test = clf_if.decision_function(X_test)

# Predict outliers using Isolation Forest
y_pred_train_if = clf_if.predict(X_train)
y_pred_test_if = clf_if.predict(X_test)

# Flag anomalies based on the Isolation Forest predictions
is_anomaly_train_if = y_pred_train_if == -1
is_anomaly_test_if = y_pred_test_if == -1
is_anomaly_if = np.concatenate((is_anomaly_train_if, is_anomaly_test_if))

df['is_anomaly_if'] = is_anomaly_if

# Fit the Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
lof.fit(X_train)

# Calculate anomaly scores for the test data using LOF
anomaly_scores_lof_train = -lof.negative_outlier_factor_
anomaly_scores_lof_test = -lof.negative_outlier_factor_

# Predict outliers using LOF
y_pred_train_lof = lof.fit_predict(X_train)
y_pred_test_lof = lof.fit_predict(X_test)

# Flag anomalies based on the LOF predictions
is_anomaly_train_lof = y_pred_train_lof == -1
is_anomaly_test_lof = y_pred_test_lof == -1
is_anomaly_lof = np.concatenate((is_anomaly_train_lof, is_anomaly_test_lof))

df['is_anomaly_lof'] = is_anomaly_lof

# Define the autoencoder model
input_dim = len(features)
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='relu')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=64, shuffle=True, validation_data=(X_test, X_test))

# Reconstruct data using autoencoder
reconstructed_data = autoencoder.predict(df[features])

# Calculate mean squared error
mse = np.mean(np.power(df[features] - reconstructed_data, 2), axis=1)

# Calculate the threshold
threshold = np.percentile(mse, 5)

# Flag anomalies based on the threshold
df['is_anomaly_autoencoder'] = mse < threshold

# Define the sequence length for LSTM
sequence_length = 10

# Create sequences for training LSTM
sequences = [X_train.values[i:i+sequence_length] for i in range(len(X_train) - sequence_length)]
X_train_lstm = np.array(sequences)
y_train_lstm = X_train.iloc[sequence_length:].values

# Define LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model_lstm.add(Dense(X_train_lstm.shape[2]))  # Output layer with same number of features as input
model_lstm.compile(optimizer='adam', loss='mse')

# Train the LSTM model
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=64, shuffle=True)

# Make predictions on the training data
y_pred_lstm = model_lstm.predict(X_train_lstm)

# Calculate reconstruction error for each data point
mse_lstm = np.mean(np.power(y_train_lstm - y_pred_lstm, 2), axis=1)

# Calculate the threshold
threshold_lstm = np.percentile(mse_lstm, 95)

# Flag anomalies based on the threshold
anomaly_flags_lstm = mse_lstm > threshold_lstm

# Padding the anomaly flags to match the length of the DataFrame index
anomaly_flags_lstm = np.pad(anomaly_flags_lstm, (sequence_length - 1, 0), mode='constant', constant_values=False)
anomaly_flags_lstm = np.concatenate((np.zeros(sequence_length - 1, dtype=bool), anomaly_flags_lstm))
df['is_anomaly_lstm'] = np.concatenate((anomaly_flags_lstm, np.zeros(len(df) - len(anomaly_flags_lstm), dtype=bool)))


# Plotting
@app.route('/')
def index():
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 20), dpi=100)

    # Scatter plot for One-Class SVM
    axes[0, 0].scatter(range(len(anomaly_scores_svm_train)), anomaly_scores_svm_train, label='Train Set', color='blue', alpha=0.5)
    axes[0, 0].scatter(range(len(anomaly_scores_svm_train), len(anomaly_scores_svm_train) + len(anomaly_scores_svm_test)), anomaly_scores_svm_test, label='Test Set', color='red', alpha=0.5)
    axes[0, 0].axhline(y=threshold_train, color='green', linestyle='--', label='Train Threshold')
    axes[0, 0].axhline(y=threshold_test, color='orange', linestyle='--', label='Test Threshold')
    axes[0, 0].set_title('Anomaly Scores (One-Class SVM)')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Anomaly Score')
    axes[0, 0].legend()

    # Histogram plot for One-Class SVM
    axes[0, 1].hist(anomaly_scores_svm_train, bins=50, alpha=0.5, color='blue', label='Train Set')
    axes[0, 1].hist(anomaly_scores_svm_test, bins=50, alpha=0.5, color='red', label='Test Set')
    axes[0, 1].axvline(x=threshold_train, color='green', linestyle='--', label='Train Threshold')
    axes[0, 1].axvline(x=threshold_test, color='orange', linestyle='--', label='Test Threshold')
    axes[0, 1].set_title('Histogram of Anomaly Scores (One-Class SVM)')
    axes[0, 1].set_xlabel('Anomaly Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # Scatter plot for Isolation Forest
    axes[1, 0].scatter(range(len(anomaly_scores_if_train)), anomaly_scores_if_train, color='blue', label='Train', alpha=0.5)
    axes[1, 0].scatter(range(len(anomaly_scores_if_train), len(anomaly_scores_if_train) + len(anomaly_scores_if_test)), anomaly_scores_if_test, color='red', label='Test', alpha=0.5)
    axes[1, 0].set_title('Anomaly Scores (Isolation Forest)')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Anomaly Score')
    axes[1, 0].legend()

    # Histogram plot for Isolation Forest
    axes[1, 1].hist(anomaly_scores_if_train, bins=50, color='blue', edgecolor='black', alpha=0.5, label='Train')
    axes[1, 1].hist(anomaly_scores_if_test, bins=50, color='red', edgecolor='black', alpha=0.5, label='Test')
    axes[1, 1].set_title('Histogram of Anomaly Scores (Isolation Forest)')
    axes[1, 1].set_xlabel('Anomaly Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    # Scatter plot for Local Outlier Factor
    axes[2, 0].scatter(range(len(anomaly_scores_lof_train)), anomaly_scores_lof_train, color='blue', label='Train', alpha=0.5)
    axes[2, 0].scatter(range(len(anomaly_scores_lof_train), len(anomaly_scores_lof_train) + len(anomaly_scores_lof_test)), anomaly_scores_lof_test, color='red', label='Test', alpha=0.5)
    axes[2, 0].set_title('Anomaly Scores (Local Outlier Factor)')
    axes[2, 0].set_xlabel('Index')
    axes[2, 0].set_ylabel('Anomaly Score')
    axes[2, 0].legend()

    # Histogram plot for Local Outlier Factor
    axes[2, 1].hist(anomaly_scores_lof_train, bins=50, color='blue', edgecolor='black', alpha=0.5, label='Train')
    axes[2, 1].hist(anomaly_scores_lof_test, bins=50, color='red', edgecolor='black', alpha=0.5, label='Test')
    axes[2, 1].set_title('Histogram of Anomaly Scores (Local Outlier Factor)')
    axes[2, 1].set_xlabel('Anomaly Score')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].legend()
    
    # Scatter plot for Autoencoder
    axes[3, 0].scatter(range(len(mse)), mse, color='blue', alpha=0.5)
    axes[3, 0].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    axes[3, 0].set_title('Reconstruction Error vs. Index (Autoencoder)')
    axes[3, 0].set_xlabel('Index')
    axes[3, 0].set_ylabel('Reconstruction Error')
    axes[3, 0].legend()

    # Histogram plot for Autoencoder
    axes[3, 1].hist(mse, bins=50, color='blue', alpha=0.7)
    axes[3, 1].axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    axes[3, 1].set_title('Histogram of Reconstruction Error (Autoencoder)')
    axes[3, 1].set_xlabel('Reconstruction Error')
    axes[3, 1].set_ylabel('Frequency')
    axes[3, 1].legend()
    
    # Scatter plot for LSTM
    axes[4, 0].scatter(range(len(mse_lstm)), mse_lstm, color='blue', alpha=0.5)
    axes[4, 0].axhline(y=threshold_lstm, color='r', linestyle='--', label='Threshold')
    axes[4, 0].set_title('Reconstruction Error vs. Index (LSTM)')
    axes[4, 0].set_xlabel('Index')
    axes[4, 0].set_ylabel('Reconstruction Error')
    axes[4, 0].legend()

    # Histogram plot for LSTM
    axes[4, 1].hist(mse_lstm, bins=50, color='blue', alpha=0.7)
    axes[4, 1].axvline(x=threshold_lstm, color='r', linestyle='--', label='Threshold')
    axes[4, 1].set_title('Histogram of Reconstruction Error (LSTM)')
    axes[4, 1].set_xlabel('Reconstruction Error')
    axes[4, 1].set_ylabel('Frequency')
    axes[4, 1].legend()

    # Save the figure to bytes
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the bytes to base64
    img = base64.b64encode(buffer.getvalue()).decode()

    # Scatter plot for Accuracy of Anomaly Detection Methods
    fig_accuracy, ax_accuracy = plt.subplots(figsize=(8, 6))
    methods = ['One-Class SVM', 'Isolation Forest', 'Local Outlier Factor', 'Autoencoder', 'LSTM']
    accuracies = [0.946195652173913, 0.8739130434782608, 0.8641304347826086, 0.9402173913043478, 0.9347826086956522]  # Given accuracies
    ax_accuracy.bar(methods, accuracies, color='blue')
    ax_accuracy.set_title('Accuracy of Anomaly Detection Methods')
    ax_accuracy.set_xlabel('Anomaly Detection Method')
    ax_accuracy.set_ylabel('Accuracy')
    ax_accuracy.set_ylim(0.8, 1)

    # Save the accuracy plot to bytes
    buffer_accuracy = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer_accuracy, format='png')
    buffer_accuracy.seek(0)

    # Encode the accuracy plot bytes to base64
    img_accuracy = base64.b64encode(buffer_accuracy.getvalue()).decode()

    return render_template('index.html', img=img, img_accuracy=img_accuracy)

if __name__ == '__main__':
    app.run(debug=True)




