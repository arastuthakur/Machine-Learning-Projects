# Support Vector Machines (SVM) Explained

## Introduction
Support Vector Machines (SVM) are powerful machine learning models used for classification and regression tasks. In this guide, we will explore SVM from fundamental concepts to advanced techniques.

## Table of Contents
1. [What is Support Vector Machines (SVM)?](#what-is-support-vector-machines-svm)
2. [Linear SVM](#linear-svm)

3. [Non-linear SVM](#non-linear-svm)

4. [Multi-class Classification](#multi-class-classification)

5. [Regression with SVM](#regression-with-svm)


9. [Resources](#resources)
   - [Books](#books)
   - [Online Courses](#online-courses)
   - [Additional Reading](#additional-reading)

## What is Support Vector Machines (SVM)?
Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification and regression tasks. SVM aims to find the hyperplane that best separates data points into different classes while maximizing the margin.
Introduction
SVM is a powerful supervised algorithm that works best on smaller datasets but on complex ones. Support Vector Machine, abbreviated as SVM can be used for both regression and classification tasks, but generally, they work best in classification problems. They were very famous around the time they were created, during the 1990s, and keep on being the go-to method for a high-performing algorithm with a little tuning.

## Linear SVM
Linear SVM is the foundation of SVM models and deals with linearly separable data.
When the data is perfectly linearly separable only then we can use Linear SVM. Perfectly linearly separable means that the data points can be classified into 2 classes by using a single straight line(if 2D).

## Non-linear SVM
When the data is not linearly separable then we can use Non-Linear SVM, which means when the data points cannot be separated into 2 classes by using a straight line (if 2D) then we use some advanced techniques like kernel tricks to classify them. In most real-world applications we do not find linearly separable datapoints hence we use kernel trick to solve them.
Support Vectors: These are the points that are closest to the hyperplane. A separating line will be defined with the help of these data points.
Margin: it is the distance between the hyperplane and the observations closest to the hyperplane (support vectors). In SVM large margin is considered a good margin. There are two types of margins hard margin and soft margin. I will talk more about these two in the later section.
https://editor.analyticsvidhya.com/uploads/567891.png

### Kernel Functions
The most interesting feature of SVM is that it can even work with a non-linear dataset and for this, we use “Kernel Trick” which makes it easier to classifies the points. Suppose we have a dataset like this:

dataset
Here we see we cannot draw a single line or say hyperplane which can classify the points correctly. So what we do is try converting this lower dimension space to a higher dimension space using some quadratic functions which will allow us to find a decision boundary that clearly divides the data points. These functions which help us do this are called Kernels and which kernel to use is purely determined by hyperparameter tuning.


Different Kernel Functions
Some kernel functions which you can use in SVM are given below:

1. Polynomial Kernel
Following is the formula for the polynomial kernel:

Formula for the polynomial kernel
Here d is the degree of the polynomial, which we need to specify manually.

Suppose we have two features X1 and X2 and output variable as Y, so using polynomial kernel we can write it as:

formula for the polynomial kernel
So we basically need to find X12 , X22 and X1.X2, and now we can see that 2 dimensions got converted into 5 dimensions.

A SVM using Polynomial kernal
 

Image 4
2. Sigmoid Kernel
We can use it as the proxy for neural networks. Equation is:

Equation for Sigmoid kernal
It is just taking your input, mapping them to a value of 0 and 1 so that they can be separated by a simple straight line.

Support Vector Classifier using Sigmoid Kernal
Image Source: https://dataaspirant.com/svm-kernels/#t-1608054630725

3. RBF Kernel
What it actually does is to create non-linear combinations of our features to lift your samples onto a higher-dimensional feature space where we can use a linear decision boundary to separate your classes It is the most used kernel in SVM classifications, the following formula explains it mathematically:

Formula for RBF kernal
where,

1. ‘σ’ is the variance and our hyperparameter
2. ||X₁ – X₂|| is the Euclidean Distance between two points X₁ and X₂


4. Bessel function kernel
It is mainly used for eliminating the cross term in mathematical functions. Following is the formula of the Bessel function kernel:

formula of the Bessel function kernel
5. Anova Kernel
It performs well on multidimensional regression problems. The formula for this kernel function is:

Formula for Anova Kernel
 https://editor.analyticsvidhya.com/uploads/1923729.2.png


## Implementing SVM
```Python
import numpy as np

# Sample data: Iris dataset (two classes, two features)
# Sepal length and sepal width for setosa and versicolor
X = np.array([[5.1, 3.5], [4.9, 3.0], [5.7, 2.8], [6.0, 3.0], [5.5, 2.4], [6.2, 2.9],
              [6.7, 3.1], [6.7, 3.0], [5.7, 2.6], [6.0, 2.2]])
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

# SVM parameters
learning_rate = 0.01
epochs = 1000
C = 1.0

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0

# Training the SVM using the simplified SMO algorithm
for epoch in range(epochs):
    for i in range(len(X)):
        condition = y[i] * (np.dot(w, X[i]) + b) >= 1
        if not condition:
            w = w - learning_rate * (w - C * y[i] * X[i])
            b = b - learning_rate * (-C * y[i])

# Calculate the margin (distance between support vectors)
margin = 2 / np.linalg.norm(w)

# Display results
print("Optimal weights:", w)
print("Optimal bias:", b)
print("Margin:", margin)
```
### Using Python and Libraries
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Split the dataset into a training set and a testing set (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)
# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)
```

## Resources
Explore further resources to deepen your understanding of SVM.

## Books

1. **"Support Vector Machines"** by Nello Cristianini and John Shawe-Taylor - A comprehensive book that covers the theory and practical aspects of SVM.

2. **"Pattern Recognition and Machine Learning"** by Christopher M. Bishop - This book includes a section on SVM and provides a broader perspective on pattern recognition and machine learning.

## Online Courses

3. **Coursera - "Machine Learning" by Andrew Ng** - This foundational machine learning course includes a section on SVM, offering in-depth insights into SVM theory and implementation.

4. **edX - "Support Vector Machines"** - A course dedicated to SVM, covering both theory and practical implementation.

## Additional Reading

5. **[Support Vector Machines for Classification and Regression](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)** - The original paper introducing the popular LIBSVM library, which is widely used for SVM implementations.

6. **[A Tutorial on Support Vector Machines for Pattern Recognition](http://www.clopinet.com/SVM.applications.html)** - A comprehensive tutorial on SVM with a focus on practical applications.

7. **[Understanding Support Vector Machines](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/svmtutorial.pdf)** - A tutorial by Microsoft Research that provides insights into SVM theory and implementation.

8. **[Support Vector Machines: A Practical Guide](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)** - A practical guide to using SVM, including tips and best practices.

## GitHub Repositories

9. **[Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/svm.html)** - The official documentation of scikit-learn provides detailed guides and examples for SVM in Python.

10. **[LIBSVM GitHub Repository](https://github.com/cjlin1/libsvm)** - The GitHub repository for the popular LIBSVM library, where you can find the source code and examples.

These resources cover a wide range of topics related to Support Vector Machines, from theory and implementation to practical applications. Whether you're a beginner or an experienced practitioner, you'll find valuable materials to deepen your understanding of this essential machine learning algorithm.
