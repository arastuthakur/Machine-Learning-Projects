Artificial Neural Networks (ANNs) are a class of machine learning models inspired by the structure and function of the human brain. They consist of interconnected nodes, or artificial neurons, organized in layers. ANNs are used for various tasks, such as classification, regression, and pattern recognition. Below, I'll provide an in-depth explanation of ANNs and an advanced Python code example using the popular deep learning library, TensorFlow.

Basic Concepts
Neuron (Node): The basic building block of an ANN. Neurons take inputs, perform a weighted sum, apply an activation function, and produce an output.

Layer: Neurons are organized into layers. An ANN typically has an input layer, one or more hidden layers, and an output layer.

Weights and Biases: Each connection between neurons has an associated weight, and each neuron has a bias. These parameters are learned during training.

Activation Function: It introduces non-linearity to the model. Common activation functions include ReLU (Rectified Linear Unit), Sigmoid, and Tanh.

Forward Propagation: The process of computing the output of the network by passing input data through the layers.

Loss Function: Measures the difference between the predicted output and the actual target values.

Backpropagation: The process of updating the weights and biases by computing gradients of the loss function with respect to the network's parameters.

Training: The iterative process of updating the network's parameters to minimize the loss function.
```python
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple feedforward neural network
model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
```
