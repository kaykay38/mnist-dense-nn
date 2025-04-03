# Handwriting Number Recognition using a Custom Dense Neural Network

## Overview
This project implements a feed-forward neural network from scratch to recognize handwritten digits using the MNIST dataset. The implementation uses stochastic gradient descent with backpropagation for training the network.

## Features
- Custom neural network implementation with configurable layers
- Support for various activation functions (tanh, logistic, ReLU, identity)
- Mini-batch stochastic gradient descent for efficient training
- High accuracy on MNIST dataset (>97% accuracy on validation set)
- Visualization of correctly and incorrectly classified digits
- Performance metrics tracking and reporting

## Project Structure
```
├── nn/
│   ├── nn.py           # Core neural network implementation
│   ├── nn_layer.py     # Neural network layer implementation
│   └── math_util.py    # Mathematical utility functions
├── nn_utils/
│   └── utils.py        # Utility functions
├── MNIST/              # MNIST dataset directory
├── output/             # Generated visualizations and metrics
├── test_mnist.py       # Main training and testing script
└── README.md          # Project documentation
```

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- tqdm

## Installation
1. Clone the repository
2. Extract the MNIST dataset from `MNIST.zip` into the `MNIST` directory
3. Install the required dependencies:
```bash
pip install numpy pandas matplotlib tqdm
```

## Usage

The main script `test_mnist.py` implements a neural network with the following architecture:
- Input Layer: 784 neurons (28x28 pixel images)
- Hidden Layer 1: 100 neurons (ReLU activation)
- Hidden Layer 2: 30 neurons (ReLU activation)
- Output Layer: 10 neurons (Logistic activation)

To run the script:
```bash
python test_mnist.py
```

To create and train a neural network:
```python
from nn.nn import NeuralNetwork

# Create a network
nn = NeuralNetwork()

# Add layers (input layer, hidden layers, output layer)
nn.add_layer(784)  # Input layer (MNIST images are 28x28 = 784 pixels)
nn.add_layer(128, 'tanh')  # Hidden layer
nn.add_layer(10, 'tanh')   # Output layer (10 digits)

# Train the network
nn.fit(X_train, y_train, eta=0.1, iterations=100000, SGD=True, mini_batch_size=20)

# Make predictions
predictions = nn.predict(X_test)
```

## Model Architecture
- Input Layer: 784 neurons (28x28 pixel images)
- Configurable hidden layers with choice of activation functions
- Output Layer: 10 neurons (one for each digit 0-9)

## Outputs
The script generates the following in the `output` directory:
- `metrics.txt`: Training and test error rates
- `misclassified.png`: Sample of misclassified digits with predicted vs actual labels
- `correctly_classified.png`: Sample of correctly classified digits with labels

## Performance
The model achieves >97% accuracy on the MNIST validation dataset when properly trained with 100000 iterations.

## Author
Mia Hunt

## License
This project is open source and available under the MIT License.
