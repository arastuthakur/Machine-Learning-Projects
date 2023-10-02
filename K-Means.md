# K-Means Clustering Algorithm Explained

## Introduction
K-Means is a popular unsupervised machine learning algorithm used for clustering data into groups or clusters. In this guide, we will explore the K-Means algorithm from fundamental concepts to advanced techniques.

## Table of Contents
1. [What is K-Means Clustering?](#what-is-k-means-clustering)
2. [How K-Means Works](#how-k-means-works)
3. [Implementing K-Means](#implementing-k-means)
   - [Using Python and Libraries](#using-python-and-libraries)
   - [Step-by-step Guide](#step-by-step-guide)
4. [Resources](#resources)
   - [Books](#books)
   - [Online Courses](#online-courses)
   - [Additional Reading](#additional-reading)

## What is K-Means Clustering?
K-Means clustering is an unsupervised machine learning algorithm used to partition a dataset into distinct groups or clusters based on similarity.
A cluster refers to a collection of data points aggregated together because of certain similarities.
You’ll define a target number k, which refers to the number of centroids you need in the dataset. A centroid is the imaginary or real location representing the center of the cluster.
Every data point is allocated to each of the clusters through reducing the in-cluster sum of squares.
In other words, the K-means algorithm identifies k number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.
The ‘means’ in the K-means refers to averaging of the data; that is, finding the centroid.
## How K-Means Works
To process the learning data, the K-means algorithm in data mining starts with a first group of randomly selected centroids, which are used as the beginning points for every cluster, and then performs iterative (repetitive) calculations to optimize the positions of the centroids
It halts creating and optimizing clusters when either:
The centroids have stabilized — there is no change in their values because the clustering has been successful.
The defined number of iterations has been achieved.
## Implementing K-Means
```python
import numpy as np

# Sample data points
data = np.array([
    [2, 3],
    [3, 3],
    [4, 3],
    [8, 8],
    [9, 8],
    [10, 8]
])

# Number of clusters (K)
K = 2

# Step 1: Initialize centroids
initial_centroids = data[np.random.choice(data.shape[0], K, replace=False)]

# Initialize centroids (step 1)
centroids = initial_centroids

# Maximum number of iterations
max_iterations = 100

for iteration in range(max_iterations):
    # Step 2: Assignment
    distances = np.sqrt(np.sum((data[:, np.newaxis] - centroids) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)
    
    # Step 3: Update centroids
    new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(K)])
    
    # Convergence check
    if np.all(centroids == new_centroids):
        break
    
    centroids = new_centroids

# Final cluster centroids
print("Final centroids:")
print(centroids)
```
### Using Python and Libraries
```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the Iris dataset from seaborn
iris = sns.load_dataset('iris')

# Extract the features (sepal_length, sepal_width, petal_length, petal_width)
X = iris.iloc[:, :-1]

# Number of clusters (K)
K = 3

# Create a K-Means model
kmeans = KMeans(n_clusters=K)

# Fit the model to the data
kmeans.fit(X)

# Get cluster labels for each data point
cluster_labels = kmeans.labels_

# Add cluster labels to the dataset
iris['cluster'] = cluster_labels

# Visualize the clusters
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='cluster', palette='Dark2')
plt.title('K-Means Clustering of Iris Dataset')
plt.show()
```
## Resources
Explore further resources to deepen your understanding of K-Means clustering.

### Books

- **"Pattern Recognition and Machine Learning"** by Christopher M. Bishop - This book provides a comprehensive introduction to clustering algorithms, including K-Means.

- **"Data Science for Business"** by Foster Provost and Tom Fawcett - A valuable resource that covers clustering techniques, including K-Means, from a business perspective.

### Online Courses

- **Coursera - "Machine Learning" by Andrew Ng** - This popular online course includes a section on clustering, which covers K-Means and other clustering algorithms.

- **edX - "Clustering and Retrieval"** - A course that focuses on clustering techniques, including K-Means, and their applications in information retrieval.

### Additional Reading

- **[K-Means Clustering (Wikipedia)](https://en.wikipedia.org/wiki/K-means_clustering)** - The Wikipedia page provides a detailed overview of K-Means clustering, including its algorithmic details.

- **[Scikit-Learn Documentation - K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)** - The official documentation of scikit-learn includes information on K-Means clustering and examples.

- **[An Introduction to Clustering and different methods of clustering](https://towardsdatascience.com/an-introduction-to-clustering-algorithms-in-python-123438574097)** - An article that introduces different clustering methods, including K-Means, with code examples.

These resources cover a wide range of topics related to K-Means clustering, from theoretical foundations to practical applications. Whether you're a beginner or an experienced practitioner, these materials can help you deepen your knowledge and skills in K-Means clustering for data analysis and machine learning.
