import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load data into pandas
data = pd.read_csv('lstm_dataset.csv')
X = data[['time_sin', 'time_cos', 'year_day_sin','year_day_cos', 'temp', 'pressure', 'humidity']]
y = data['temp']

# Normalize all features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Create sequences
def create_sequences(X, y, sequence_length):
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:(i + sequence_length)])  # Extract the sequence of features
        ys.append(y[i + sequence_length])  # Get the temperature at the next hour
    return np.array(Xs), np.array(ys)

sequence_length = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, shuffle=False)

# Convert to TensorFlow dataset
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Batch data
train_data = train_data.batch(32).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

# list of model configurations to be tested
model_configs = [
    {'lstm_units': [50, 25], 'dense_units': [25], 'dropout_rate': 0.2, 'learning_rate': 0.0005},
    {'lstm_units': [100, 50], 'dense_units': [50], 'dropout_rate': 0.3, 'learning_rate': 0.0003},
    {'lstm_units': [75, 50, 25], 'dense_units': [50, 25], 'dropout_rate': 0.25, 'learning_rate': 0.0001},
    {'lstm_units': [25], 'dense_units': [10], 'dropout_rate': 0.1, 'learning_rate': 0.001},
    {'lstm_units': [50], 'dense_units': [20], 'dropout_rate': 0.15, 'learning_rate': 0.0005},
    {'lstm_units': [200, 100, 50], 'dense_units': [100, 50], 'dropout_rate': 0.35, 'learning_rate': 0.00005},
    {'lstm_units': [150, 75], 'dense_units': [75], 'dropout_rate': 0.3, 'learning_rate': 0.0002},
    {'lstm_units': [100, 50, 25, 12], 'dense_units': [50, 25], 'dropout_rate': 0.25, 'learning_rate': 0.0001},
]



# Create results file
results_file = 'lstm_results.csv'
results_columns = ['model_path', 'model_architecture', 'test_loss']
results_df = pd.DataFrame(columns=results_columns)

# Loop through each model configuration
for i, config in enumerate(model_configs):
    print(f"Training model {i+1} with configuration: {config}")

    # Build the model
    model = Sequential()
    for idx, units in enumerate(config['lstm_units']):
        if idx == 0:
            model.add(LSTM(units, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])))
        else:
            model.add(LSTM(units, return_sequences=(idx < len(config['lstm_units']) - 1)))
        model.add(Dropout(config['dropout_rate']))
    
    for units in config['dense_units']:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(config['dropout_rate']))

    model.add(Dense(1))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(train_data, epochs=20, validation_data=test_data, callbacks=[early_stopping, reduce_lr])

    # Save the model
    model_path = f'model_{i+1}.keras'
    model.save(model_path)

    print(f"Training complete and model saved to '{model_path}'.")

    # Make predictions on the test set
    predictions = model.predict(test_data)

    # Inverse transform the predictions and actual values
    predictions_original_scale = scaler_y.inverse_transform(predictions)
    y_test_original_scale = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # Compare predictions to actual values
    comparison = pd.DataFrame({
        'Actual': y_test_original_scale.flatten(),
        'Predicted': predictions_original_scale.flatten()
    })

    # Display the comparison DataFrame
    print(comparison.head())

    # Evaluate the model
    test_loss = model.evaluate(test_data)
    print(f"Test Loss for model {i+1}: {test_loss}")

    # Record the results
    model_architecture = f"LSTM Units: {config['lstm_units']}, Dense Units: {config['dense_units']}, Dropout Rate: {config['dropout_rate']}, Learning Rate: {config['learning_rate']}"
    results_df = results_df.append({
        'model_path': model_path,
        'model_architecture': model_architecture,
        'test_loss': test_loss
    }, ignore_index=True)

# Save results to CSV
results_df.to_csv(results_file, index=False)
print(f"Results saved to '{results_file}'.")

print("All models trained and evaluated.")
