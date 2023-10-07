# Artificial Neural Networks (ANNs) Explained

## Introduction
Artificial Neural Networks (ANNs) are at the core of modern machine learning and deep learning. They are designed to mimic the structure and functioning of the human brain to solve complex problems. In this comprehensive guide, we will delve into ANNs, covering their principles, architectures, training techniques, and real-world applications.

## Table of Contents
1. [Introduction to Artificial Neural Networks](#introduction-to-artificial-neural-networks)
2. [Neurons and Layers](#neurons-and-layers)
   - [Neurons as Building Blocks](#neurons-as-building-blocks)
   - [Layers in ANNs](#layers-in-anns)
   - [Activation Functions](#activation-functions)
3. [Implementing ANNs](#implementing-anns)
   - [Using Deep Learning Frameworks](#using-deep-learning-frameworks)
4. [Resources](#resources)
    - [Books](#books)
    - [Online Courses](#online-courses)
    - [Additional Reading](#additional-reading)

## Introduction to Artificial Neural Networks
Artificial Neural Networks (ANN) are algorithms based on brain function and are used to model complicated patterns and forecast issues. The Artificial Neural Network (ANN) is a deep learning method that arose from the concept of the human brain Biological Neural Networks. The development of ANN was the result of an attempt to replicate the workings of the human brain. The workings of ANN are extremely similar to those of biological neural networks, although they are not identical. ANN algorithm accepts only numeric and structured data.
Convolutional Neural Networks (CNN) and Recursive Neural Networks (RNN) are used to accept unstructured and non-numeric data forms such as Image, Text, and Speech. This article focuses solely on Artificial Neural Networks.
## Neurons and Layers
Number of Neurons In Input and Output Layers
-The number of neurons in the input layer is equal to the number of features in the data and in very rare cases, there will be one input layer for bias. Whereas the number of neurons in the output depends on whether is the model is used as a regressor or classifier. If the model is a regressor then the output layer will have only a single neuron but in case if the model is a classifier it will have a single neuron or multiple neurons depending on the class label of the model.
Number of Neurons and Number of Layers in Hidden Layer
-When it comes to the hidden layers, the main concerns are how many hidden layers and how many neurons are required?
-An Introduction to Neural Networks for Java, Second Edition by jeffheaton it is mentioned that number of hidden layers is determined as below.
<img width="453" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/27b878d0-036f-428b-b530-1a7fdb247ba7">
There are many rule-of-thumb methods for determining the correct number of neurons to use in the hidden layers, such as the following:
1.The number of hidden neurons should be between the size of the input layer and the size of the output layer.
2.The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
3.The number of hidden neurons should be less than twice the size of the input layer.
Moreover, the number of neurons and number layers required for the hidden layer also depends upon training cases, amount of outliers, the complexity of, data that is to be learned, and the type of activation functions used.
Most of the problems can be solved by using a single hidden layer with the number of neurons equal to the mean of the input and output layer. If less number of neurons is chosen it will lead to underfitting and high statistical bias. Whereas if we choose too many neurons it may lead to overfitting, high variance, and increases the time it takes to train the network.
### Neurons as Building Blocks
Artificial neurons, often referred to as "perceptrons," are fundamental building blocks of artificial neural networks (ANNs). They serve as the basic processing units in these networks, and understanding their structure and function is crucial to grasp the foundations of deep learning and machine learning.

### Structure

An artificial neuron consists of the following components:

1. **Inputs**: Neurons receive multiple input signals, which are typically real numbers representing features or activations from preceding neurons in the network. Each input is associated with a weight that represents its importance.

2. **Weights**: Weights are parameters that modulate the strength of the input signals. They determine how much influence each input has on the neuron's output. These weights are adjusted during the training process to optimize the network's performance.

3. **Summation Function**: The neuron computes a weighted sum of its inputs. This summation is a linear combination of the input values and their corresponding weights.

4. **Activation Function**: The weighted sum is passed through an activation function. This function introduces non-linearity into the neuron's response, allowing the neuron to model complex relationships in the data. Common activation functions include the sigmoid, ReLU (Rectified Linear Unit), and tanh (hyperbolic tangent).

5. **Output**: The output of the neuron is the result of applying the activation function to the weighted sum. This output is then passed to other neurons in the network if it is part of a hidden layer, or it serves as the final output of the network if it is an output neuron.

### Function

The primary function of an artificial neuron is to perform a weighted sum of its inputs and produce an output based on this summation. Here's a step-by-step explanation of its function:

1. **Input Combination**: The neuron receives multiple input signals, each multiplied by its associated weight. These weighted inputs are summed together.

2. **Summation**: The neuron calculates the sum of the weighted inputs. Mathematically, this can be represented as:

## Structure and Function of Artificial Neurons

Artificial neurons, often referred to as "perceptrons," are fundamental building blocks of artificial neural networks (ANNs). They serve as the basic processing units in these networks, and understanding their structure and function is crucial to grasp the foundations of deep learning and machine learning.

### Structure

An artificial neuron consists of the following components:

1. **Inputs**: Neurons receive multiple input signals, which are typically real numbers representing features or activations from preceding neurons in the network. Each input is associated with a weight that represents its importance.

2. **Weights**: Weights are parameters that modulate the strength of the input signals. They determine how much influence each input has on the neuron's output. These weights are adjusted during the training process to optimize the network's performance.

3. **Summation Function**: The neuron computes a weighted sum of its inputs. This summation is a linear combination of the input values and their corresponding weights.

4. **Activation Function**: The weighted sum is passed through an activation function. This function introduces non-linearity into the neuron's response, allowing the neuron to model complex relationships in the data. Common activation functions include the sigmoid, ReLU (Rectified Linear Unit), and tanh (hyperbolic tangent).

5. **Output**: The output of the neuron is the result of applying the activation function to the weighted sum. This output is then passed to other neurons in the network if it is part of a hidden layer, or it serves as the final output of the network if it is an output neuron.

### Function

The primary function of an artificial neuron is to perform a weighted sum of its inputs and produce an output based on this summation. Here's a step-by-step explanation of its function:

1. **Input Combination**: The neuron receives multiple input signals, each multiplied by its associated weight. These weighted inputs are summed together.

2. **Summation**: The neuron calculates the sum of the weighted inputs. Mathematically, this can be represented as:

Sum = (w1 * input1) + (w2 * input2) + ... + (wn * inputn)
3. **Activation**: The neuron applies an activation function to the computed sum. This introduces non-linearity into the neuron's response and helps the neuron learn complex patterns in the data.

4. **Output**: The result of the activation function becomes the neuron's output. This output is then used as an input to other neurons in the network, propagating information through the network's layers.

### Layers in ANNs
## Input, Hidden, and Output Layers in Artificial Neural Networks (ANNs)

Artificial Neural Networks (ANNs) are composed of multiple layers that work together to process and transform input data into meaningful output. These layers are categorized into three main types: input, hidden, and output layers.

### Input Layer

The **input layer** is the first layer in an ANN and serves as the entry point for the network. Key characteristics of the input layer include:

- **Neurons**: Each neuron in the input layer represents a feature or an input variable. The number of neurons in this layer is determined by the dimensionality of the input data. For example, in an image classification task, each neuron in the input layer may correspond to a pixel in the input image.

- **No Processing**: Neurons in the input layer do not perform any computation. They simply pass the input data to the subsequent layers. The input values are usually normalized or standardized before being fed into the network to ensure consistent processing.

- **Direct Connections**: Each neuron in the input layer is connected to every neuron in the next layer, known as the first hidden layer. These connections have associated weights that determine the strength of the connections.

### Hidden Layer(s)

Hidden layers are intermediate layers in the ANN that perform the actual computation and transformation of data. Key characteristics of hidden layers include:

- **Neurons**: Each neuron in a hidden layer processes the information received from the previous layer and produces an output. The number of hidden layers and the number of neurons in each hidden layer are design choices that can significantly impact the network's capacity to learn complex patterns.

- **Computation**: Neurons in hidden layers apply a weighted sum to their inputs, followed by the application of an activation function. This process introduces non-linearity into the network, allowing it to learn and represent complex relationships in the data.

- **Multiple Hidden Layers**: ANNs can have multiple hidden layers, forming a deep neural network (DNN). Deep networks are capable of learning hierarchical representations of data, making them suitable for tasks like image recognition and natural language processing.

### Output Layer

The **output layer** is the final layer of the ANN and produces the network's output or prediction. Key characteristics of the output layer include:

- **Neurons**: The number of neurons in the output layer depends on the nature of the task. For example, in a binary classification problem, there may be one neuron in the output layer to produce a binary decision. In multi-class classification, the number of neurons corresponds to the number of classes.

- **Activation Function**: The choice of activation function in the output layer depends on the task. For binary classification, a sigmoid or softmax activation function is often used. For regression tasks, a linear activation function may be appropriate.

- **Prediction**: The output of the output layer represents the final prediction or inference made by the network. For classification tasks, the neuron with the highest activation value typically determines the predicted class. For regression, the output is a continuous value.

### Activation Functions
![18029activation_function](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/a6852a24-87ab-4141-a600-ca1d53ebbdb8)
Activation functions play a critical role in artificial neural networks (ANNs) by introducing non-linearity into the network's computations. They determine whether a neuron should be activated (fire) or not based on the weighted sum of its inputs. Two commonly used activation functions are the Rectified Linear Unit (ReLU) and the sigmoid function.
### Rectified Linear Unit (ReLU)
The Rectified Linear Unit (ReLU) is a piecewise linear function that has gained widespread popularity in deep learning due to its simplicity and effectiveness. It is defined as follows:
f(x) = max(0, x)

Key characteristics of ReLU:

- **Non-linearity**: ReLU is a non-linear activation function. For input values less than zero, it outputs zero; for positive input values, it outputs the input value itself.

- **Efficiency**: ReLU is computationally efficient and easy to implement. It helps mitigate the vanishing gradient problem during training.

- **Sparsity**: ReLU activation can lead to sparse activations in neural networks, as many neurons may remain inactive (output zero) for certain inputs.

- **Derivative**: The derivative of ReLU is straightforward:
f'(x) = 1, if x > 0
f'(x) = 0, if x <= 0
This simplicity makes backpropagation and gradient descent easier to compute.

### Sigmoid Function

The sigmoid function, also known as the logistic function, is another commonly used activation function. It has an S-shaped curve and is defined as:

f(x) = 1 / (1 + e^(-x))

Key characteristics of the sigmoid function:

- **Range**: The sigmoid function outputs values between 0 and 1, which can be interpreted as probabilities. It is often used in binary classification problems where the output represents the probability of a positive class.

- **Smoothness**: The sigmoid function is smooth and differentiable everywhere, making it suitable for gradient-based optimization methods.

- **Vanishing Gradient**: Sigmoid can suffer from the vanishing gradient problem, especially in deep networks. This can slow down training and limit the model's ability to capture complex patterns.

- **Derivative**: The derivative of the sigmoid function is given by:
f'(x) = f(x) * (1 - f(x))
This derivative is used in backpropagation to update weights during training.

### Choosing an Activation Function

The choice of activation function depends on the nature of the problem and the architecture of the neural network. In practice:

- ReLU is often the default choice for hidden layers due to its simplicity and effectiveness, especially in deep networks (e.g., Convolutional Neural Networks).

- Sigmoid is commonly used in the output layer of binary classification problems, where the goal is to produce a probability score between 0 and 1.

- Other variants of ReLU, such as Leaky ReLU and Parametric ReLU (PReLU), address some of the limitations of standard ReLU.

Selecting the right activation function is a crucial part of designing and training neural networks, and experimentation is often necessary to determine the most suitable one for a given task.

## Implementing ANNs
```python
import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the input dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define the output dataset
y = np.array([[0], [1], [1], [0]])

# Define the number of input, hidden, and output neurons
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# Initialize weights with random values
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Set the learning rate
learning_rate = 0.1

# Set the number of training iterations
epochs = 10000

# Training the neural network
for _ in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)
    
    # Calculate the error
    error = y - output_layer_output
    
    # Backpropagation
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

# Print the final output
print("Output after training:")
print(output_layer_output)
```

### Using Deep Learning Frameworks
```python
import tensorflow as tf
import numpy as np

# Define the XOR input dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

# Define the XOR output dataset
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print("Final Loss:", loss)
print("Final Accuracy:", accuracy)

# Make predictions
predictions = model.predict(X)
print("Predictions:")
print(predictions)
```
## Resources
Find additional resources to enhance your understanding of ANNs.

### Books

- **"Deep Learning"** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville - A comprehensive book on deep learning, covering ANN fundamentals and advanced topics.

- **"Neural Networks and Deep Learning: A Textbook"** by Charu Aggarwal - A textbook that provides in-depth coverage of ANNs and their applications.

- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron - This book offers practical insights into building and training ANNs using popular libraries.

### Online Courses

- **Coursera - "Deep Learning Specialization"** by Andrew Ng - This specialization includes courses on deep learning, neural networks, and their applications.

- **edX - "Deep Learning Fundamentals"** - An edX course that covers the fundamentals of deep learning, including ANNs.

- **Coursera - "Introduction to Artificial Neural Networks and Deep Learning"** - A beginner-friendly course that introduces ANNs and deep learning concepts.

### Additional Reading

- **[Neural Networks and Deep Learning (Online Book)](http://neuralnetworksanddeeplearning.com/)** by Michael Nielsen - An online book that provides a detailed introduction to neural networks and deep learning.

- **[Deep Learning (Nature)](https://www.nature.com/articles/nature14539)** - This article in Nature provides an overview of deep learning, including ANNs, and their impact on various fields.

- **[A Gentle Introduction to Deep Learning (MIT News)](http://news.mit.edu/2017/explained-neural-networks-deep-learning-0414)** - An article that offers a gentle introduction to deep learning and neural networks.

These resources cover a wide range of topics related to Artificial Neural Networks (ANNs) and deep learning. Whether you're new to ANNs or looking to deepen your knowledge, these materials can be valuable for your journey into the world of deep learning.
