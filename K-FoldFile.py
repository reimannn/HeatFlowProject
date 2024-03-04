import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Define hyperparameters for the model
node_count = 64  # Number of nodes in each hidden layer
activation_choice = "relu"  # Activation function used in hidden layers

optimizer_choice = "rmsprop"  # Optimizer used for gradient descent
loss_choice = "mse"  # Loss function used to measure the error
metrics_choice = ["mae"]  # Metrics used to evaluate the model's performance

k = 4  # Number of folds for k-fold cross-validation
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

# Calculate the size of the training dataset and split off 20% for testing
train_data_size = sum(1 for _ in dataset)
train_size = int(1 * train_data_size)

# Process and vectorize the data for training
x_train, y_train = [], []

for x, y in dataset:
    x_train.append(x)
    y_train.append(y)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Normalize the training data
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std

# Define a function to build the neural network model
def build_model():
    model = keras.Sequential([
        layers.Dense(node_count, activation=activation_choice),
        layers.Dense(node_count, activation=activation_choice),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizer_choice, loss=loss_choice, metrics=metrics_choice)
    return model

# Perform k-fold cross-validation
num_val_samples = train_size // k
all_mae_histories = []
for i in range(k):
    print(f"Processing Fold #{i}")
    val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    partial_x_train = np.concatenate(
        [x_train[:i * num_val_samples],
         x_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_y_train = np.concatenate(
        [y_train[:i * num_val_samples],
         y_train[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_x_train, partial_y_train, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=batch_size_choice, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

# Compute the average MAE across all folds for each epoch
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# Plot the validation MAE over epochs
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# Plot the validation MAE over epochs, excluding the first few epochs for better visualization
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()
