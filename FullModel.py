import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the percentage of data to be used for testing
percentage_for_tests = .2

# Define hyperparameters for the model
node_count = 64  # Number of nodes in each hidden layer
activation_choice = "relu"  # Activation function used in hidden layers

optimizer_choice = "rmsprop"  # Optimizer used for gradient descent
loss_choice = "mse"  # Loss function used to measure the error
metrics_choice = ["mae"]  # Metrics used to evaluate the model's performance

num_epochs = 500  # Number of training epochs
batch_size_choice = 100  # Batch size used during training

# Define a function to parse the data
def parse_data(pair):
    lines = tf.strings.split(pair, '\n')
    input_parts = tf.strings.split(lines[0], ',')
    inputs = tf.strings.to_number(input_parts.values, out_type=tf.float32)
    output = tf.strings.to_number(lines[1], out_type=tf.float32)
    return inputs, output

# Set the file path for the data
file_path = "HeatData/collected.txt"

# Create a TensorFlow Dataset from the file and batch the lines in pairs
dataset = tf.data.TextLineDataset(file_path).batch(2).map(parse_data)

# Calculate the size of the training dataset and split off a portion for testing
train_data_size = sum(1 for _ in dataset)
train_size = int((1 - percentage_for_tests) * train_data_size)

# Split the data into training and testing datasets
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Process and vectorize the data for training and testing
x_train, y_train = [], []
x_test, y_test = [], []

# Extract data from the training dataset
for x, y in train_dataset:
    x_train.append(x)
    y_train.append(y)

# Extract data from the testing dataset
for x, y in test_dataset:
    x_test.append(x)
    y_test.append(y)

# Convert data to NumPy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Normalize the training and testing data
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Define a function to build the neural network model
def build_model():
    model = keras.Sequential([
        layers.Dense(node_count, activation=activation_choice),
        layers.Dense(node_count, activation=activation_choice),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizer_choice, loss=loss_choice, metrics=metrics_choice)
    return model

# Build the model
model = build_model()

# Train the model on the training data
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size_choice, verbose=0)

# Evaluate the model on the testing data
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)

# Print the Mean Absolute Error (MAE) score on the testing data
print(test_mae_score)