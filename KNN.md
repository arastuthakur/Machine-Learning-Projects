# k-Nearest Neighbors (KNN) Explained

## Introduction
k-Nearest Neighbors (KNN) is a versatile and intuitive machine learning algorithm used for classification and regression tasks. In this guide, we will explore KNN from fundamental concepts to advanced techniques.

## Table of Contents
1. [What is k-Nearest Neighbors (KNN)?](#what-is-k-nearest-neighbors-knn)
2. [How KNN Works](#how-knn-works)
   - [Distance Metrics](#distance-metrics)
   - [Choosing the Value of k](#choosing-the-value-of-k)
   - [Weighted KNN](#weighted-knn)
3. [Implementing KNN](#implementing-knn)
   - [Using Python and Libraries](#using-python-and-libraries)
   - [Step-by-step Guide](#step-by-step-guide)
4. [Resources](#resources)
   - [Books](#books)
   - [Online Courses](#online-courses)
   - [Additional Reading](#additional-reading)

## What is k-Nearest Neighbors (KNN)?
K-Nearest Neighbors (KNN) is a simple yet effective machine learning algorithm that makes predictions based on the majority class among its k-nearest neighbors in a feature space.
The K-Nearest Neighbor (KNN) algorithm is a popular machine learning technique used for classification and regression tasks. It relies on the idea that similar data points tend to have similar labels or values.

During the training phase, the KNN algorithm stores the entire training dataset as a reference. When making predictions, it calculates the distance between the input data point and all the training examples, using a chosen distance metric such as Euclidean distance.

Next, the algorithm identifies the K nearest neighbors to the input data point based on their distances. In the case of classification, the algorithm assigns the most common class label among the K neighbors as the predicted label for the input data point. For regression, it calculates the average or weighted average of the target values of the K neighbors to predict the value for the input data point.

## How KNN Works
Let’s take a simple case to understand this algorithm. Following is a spread of red circles (RC) and green squares (GS):
<img width="475" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/57b04733-7ccf-4ec0-8787-a679fde0bd4a">
You intend to find out the class of the blue star (BS). BS can either be RC or GS and nothing else. The “K” in KNN algorithm is the nearest neighbor we wish to take the vote from. Let’s say K = 3. Hence, we will now make a circle with BS as the center just as big as to enclose only three data points on the plane. Refer to the following diagram for more details:
<img width="476" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/8d1596b9-edc3-4ea0-b98f-265fadbc6073">
The three closest points to BS are all RC. Hence, with a good confidence level, we can say that the BS should belong to the class RC. Here, the choice became obvious as all three votes from the closest neighbor went to RC. The choice of the parameter K is very crucial in this algorithm. Next, we will understand the factors to be considered to conclude the best K.


### Distance Metrics
Euclidean Distance: Euclidean distance is calculated as the square root of the sum of the squared differences between a new point (x) and an existing point (y).

Manhattan Distance: This is the distance between real vectors using the sum of their absolute difference.
<img width="257" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/f33d17a1-7c35-498a-9c92-c41d49df5974">
Hamming Distance: It is used for categorical variables. If the value (x) and the value (y) are the same, the distance D will be equal to 0 . Otherwise D=1.
<img width="219" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/41fdc24d-ad47-4106-a56a-ed11504a67e7">



### Choosing the Value of k
There are no pre-defined statistical methods to find the most favorable value of K.
Initialize a random K value and start computing.
Choosing a small value of K leads to unstable decision boundaries.
The substantial K value is better for classification as it leads to smoothening the decision boundaries.
Derive a plot between error rate and K denoting values in a defined range. Then choose the K value as having a minimum error rate.

### Weighted KNN
In weighted kNN, the nearest k points are given a weight using a function called as the kernel function. The intuition behind weighted kNN, is to give more weight to the points which are nearby and less weight to the points which are farther away. Any function can be used as a kernel function for the weighted knn classifier whose value decreases as the distance increases. The simple function which is used is the inverse distance function.


## KNN for Classification
KNN can be used for both classification and regression predictive problems. However, it is more widely used in classification problems in the industry. To evaluate any technique, we generally look at 3 important aspects:

1. Ease of interpreting output

2. Calculation time

3. Predictive Power

Let us take a few examples to  place KNN in the scale :
<img width="419" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/f3807c04-4913-48e1-8160-860e9a45f30b">
KNN classifier fairs across all parameters of consideration. It is commonly used for its ease of interpretation and low calculation time.

## Implementing KNN
```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
def knn(X_train, y_train, x_test, k):
    distances = [np.sqrt(np.sum((x_train - x_test) ** 2)) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]
# Split the dataset into a training set and a testing set (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Make predictions using k-NN
k = 3  # Adjust the value of k as needed
y_pred = [knn(X_train, y_train, x_test, k) for x_test in X_test]
4. Evaluate the Model
python
Copy code
# Evaluate the model's performance
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
```
### Using Python and Libraries
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Split the dataset into a training set and a testing set (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the k-NN classifier
k = 3  # Adjust the value of k as needed
knn_classifier = KNeighborsClassifier(n_neighbors=k)
# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)
# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)
# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)
```


## Resources
Explore further resources to deepen your understanding of KNN.

## Books

1. **"Pattern Recognition and Machine Learning"** by Christopher M. Bishop - This book includes a section on k-NN and provides a broader perspective on pattern recognition and machine learning.

2. **"Introduction to the k-Nearest Neighbor Algorithm"** by Alpaydin, E. - A concise introduction to k-NN, suitable for beginners.

## Online Courses

3. **Coursera - "Machine Learning" by Andrew Ng** - This foundational machine learning course includes a section on k-NN, offering in-depth insights into the algorithm's theory and practical implementation.

4. **edX - "Introduction to Artificial Intelligence"** - A course that introduces k-NN and other machine learning techniques, suitable for beginners.

## Additional Reading

5. **[k-Nearest Neighbors Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)** - The Wikipedia page provides a detailed overview of the k-NN algorithm, including its variants and applications.

6. **[An Introduction to k-Nearest Neighbors](https://towardsdatascience.com/an-introduction-to-k-nearest-neighbors-1551b56577b4)** - A comprehensive article explaining the k-NN algorithm, its implementation, and use cases.

7. **[Understanding the k-Nearest Neighbors Algorithm](https://towardsdatascience.com/understanding-the-k-nearest-neighbors-algorithm-9b3c6320b9f5)** - An in-depth article that delves into the theory and practical aspects of k-NN.

8. **[k-Nearest Neighbors (k-NN) Algorithm Explained](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)** - A detailed explanation of k-NN with examples and code snippets.

## GitHub Repositories

9. **[Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/neighbors.html)** - The official documentation of scikit-learn provides detailed guides and examples for k-NN in Python.

10. **[K-Nearest Neighbors Implementation](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/k_nearest_neighbors.py)** - A GitHub repository containing a Python implementation of k-NN from scratch.


This comprehensive guide will equip you with the knowledge and skills to master k-Nearest Neighbors (KNN), from the basics to advanced applications in classification, regression, and more.

