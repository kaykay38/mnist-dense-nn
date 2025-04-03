import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from nn import nn
import sys

# Ensure utility functions are accessible
sys.path.append('..')
from nn_utils.utils import MyUtils

# Constants
K = 10  # Number of output classes (digits 0-9)
D = 784  # Number of input features (28x28 images)
output_folder = './output'  # Folder for saving images and metrics

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

def load_csv(file_path):
    """Load a CSV file and return as a NumPy array."""
    return pd.read_csv(file_path, header=None).to_numpy()

# Load MNIST dataset
X_train_raw = load_csv('./MNIST/x_train.csv')
y_train_raw = load_csv('./MNIST/y_train.csv')
X_test_raw = load_csv('./MNIST/x_test.csv')
y_test_raw = load_csv('./MNIST/y_test.csv')

# Dataset sizes
N_train, N_test = X_train_raw.shape[0], X_test_raw.shape[0]
print(f"Training set size: {N_train}, Test set size: {N_test}")

# Normalize all features to the range [0,1]
X_all = MyUtils.normalize_0_1(np.vstack((X_train_raw, X_test_raw)))
X_train, X_test = X_all[:N_train], X_all[N_train:]

# One-hot encoding
def one_hot_encode(y, num_classes):
    """Convert label array into one-hot encoded format."""
    return np.eye(num_classes)[y.astype(int).reshape(-1)]

y_train = one_hot_encode(y_train_raw, K)
y_test = one_hot_encode(y_test_raw, K)

# Initialize neural network
neural_net = nn.NeuralNetwork()

# Define layers
neural_net.add_layer(d=D)  # Input layer
neural_net.add_layer(d=100, act='relu')  # Hidden layer 1
neural_net.add_layer(d=30, act='relu')  # Hidden layer 2
neural_net.add_layer(d=K, act='logis')  # Output layer for classification

print("Neural network created.")

# Train the network using Stochastic Gradient Descent
print("Training the model...this may take a while...")
neural_net.fit(X_train, y_train, eta=0.1, iterations=100000, SGD=True, mini_batch_size=20)

# Calculate training and test errors
train_error = neural_net.error(X_train, y_train)
test_error = neural_net.error(X_test, y_test)

# Print and save results
print(f"Training Error: {train_error:.4f}")
print(f"Test Error: {test_error:.4f}")
with open(os.path.join(output_folder, 'metrics.txt'), 'w') as f:
    f.write(f"Training Error: {train_error:.4f}\n")
    f.write(f"Test Error: {test_error:.4f}\n")

# Get predictions
print("Classifying test sample...")
preds = neural_net.predict(X_test)


# Identify correctly and incorrectly classified samples
correct_indices = np.where(preds == y_test_raw)[0]
misclassified_indices = np.where(preds != y_test_raw)[0]

# Calculate misclassification stats
misclassified_count = len(misclassified_indices)
correct_count = len(correct_indices)
total_count = preds.shape[0]

misclassified_percentage = (misclassified_count / total_count) * 100
correct_percentage = (correct_count / total_count) * 100

# Label with stats
misclassified_label = f"Misclassified: {misclassified_count}/{total_count} ({misclassified_percentage:.2f}%)"
correct_label = f"Correctly Classified: {correct_count}/{total_count} ({correct_percentage:.2f}%)"

print(misclassified_label)
print(correct_label)

# Collect images for misclassified and correctly classified
misclassified_images = []
correct_images = []

for i in misclassified_indices[:10]:  # Choose up to 10 misclassified images
    misclassified_images.append(X_test_raw[i].reshape(28, 28))

for i in correct_indices[:10]:  # Choose up to 10 correct images
    correct_images.append(X_test_raw[i].reshape(28, 28))

def plot_classification_results(images, indices, preds, y_true, title, filename):
    """Save 5 classification results as a single 5x1 horizonal grid png image with labels."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # Wider figure for horizontal layout
    
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f"Pred: {preds[indices[i]]}\nActual: {y_true[indices[i], 0]}", pad=10)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=12, y=1.1)  # Adjust title position
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')  # Ensure everything fits
    plt.close()

# Plot misclassified images (up to 5)
plot_classification_results(
    images=misclassified_images[:5],
    indices=misclassified_indices[:5],
    preds=preds,
    y_true=y_test_raw,
    title=misclassified_label,
    filename='misclassified.png'
)

# Plot correctly classified images (up to 5)
plot_classification_results(
    images=correct_images[:5],
    indices=correct_indices[:5],
    preds=preds,
    y_true=y_test_raw,
    title=correct_label,
    filename='correctly_classified.png'
)

print(f"Prediction result images have been saved to \"{output_folder}\" folder")