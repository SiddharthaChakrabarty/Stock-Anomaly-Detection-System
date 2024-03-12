# Stock Anomaly Detection System

This project aims to detect anomalies in daily stock prices using various anomaly detection techniques and machine learning algorithms. The system retrieves historical stock price data, preprocesses it, applies anomaly detection methods, and provides insights into detected anomalies.

## Table of Contents

- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

  ## Usage

1. **Data Retrieval**: Retrieve historical stock price data using Alpha Vantage API.
2. **Preprocessing**: Clean the data and scale features using standardization.
3. **Anomaly Detection**:
   - SVM (One-Class Support Vector Machine)
   - Isolation Forest
   - Local Outlier Factor
   - Autoencoder
   - LSTM (Long Short-Term Memory)
   - KNN (K-Nearest Neighbors)
4. **Visualization**: Visualize anomaly scores and reconstruction errors.
5. **Evaluation**: Evaluate the performance of each anomaly detection method.
6. **Output**: Generate CSV files containing daily stock prices and detected anomalies.

## Dependencies

- Python 3.x
- Alpha Vantage API
- Pandas
- Matplotlib
- Scikit-learn
- Keras (with TensorFlow backend)

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or create a pull request on GitHub.
