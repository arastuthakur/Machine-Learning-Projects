# Principal Component Analysis (PCA) Explained

## Introduction
Principal Component Analysis (PCA) is a powerful dimensionality reduction technique used in various fields, including data analysis, machine learning, and image processing. In this comprehensive guide, we will explore PCA from its basic principles to advanced applications, helping you understand its significance and how to apply it effectively.
The Principal Component Analysis is a popular unsupervised learning technique for reducing the dimensionality of large data sets. It increases interpretability yet, at the same time, it minimizes information loss. It helps to find the most significant features in a dataset and makes the data easy for plotting in 2D and 3D. PCA helps in finding a sequence of linear combinations of variables.
<img width="331" alt="PrincipalComponents" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/b5bba56f-c008-476f-b6a4-7a50a0383ac6">
In the above figure, we have several points plotted on a 2-D plane. There are two principal components. PC1 is the primary principal component that explains the maximum variance in the data. PC2 is another principal component that is orthogonal to PC1.
## Table of Contents
1. [Introduction to Principal Component Analysis](#introduction-to-principal-component-analysis)
2. [Theoretical Foundations](#theoretical-foundations)
   - [Variance and Covariance](#variance-and-covariance)
   - [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
   - [Covariance Matrix](#covariance-matrix)
3. [How PCA Works](#how-pca-works)
   - [Step 1: Data Standardization](#step-1-data-standardization)
   - [Step 2: Eigendecomposition](#step-2-eigendecomposition)
   - [Step 3: Selecting Principal Components](#step-3-selecting-principal-components)
   - [Step 4: Projecting Data](#step-4-projecting-data)
4. [Implementing PCA](#implementing-pca)
   - [Using Python and Libraries](#using-python-and-libraries)
   - [Step-by-step Guide](#step-by-step-guide)
5. [Resources](#resources)
    - [Books](#books)
    - [Online Courses](#online-courses)
    - [Additional Reading](#additional-reading)

## Introduction to Principal Component Analysis
The Principal Components are a straight line that captures most of the variance of the data. They have a direction and magnitude. Principal components are orthogonal projections (perpendicular) of data onto lower-dimensional space.
## Theoretical Foundations
Principal Component Analysis (PCA) is one of the feature extraction methods to identify patterns in data, and expressing the data in such a way as to highlight their similarities and differences. One of the main advantage of PCA is that once these patterns are found in the data, the data can be compressed (i.e. the number of dimensions can be reduced) without much loss of information. This method also solves the problem of correlation among the variables.
### Variance and Covariance
1. Variance:
Variance is a measure of the spread or dispersion of data points in a dataset along a particular axis or direction. In the context of PCA, variance represents the amount of information contained in each dimension (or feature) of the dataset. The goal of PCA is to find new axes (principal components) along which the variance of the data is maximized. These principal components capture the most significant information in the data while reducing the dimensionality.
In mathematical terms, for a single dimension (variable) represented by X, the variance is calculated as:
Var(X) = (1 / (n-1)) * ∑(i=1 to n) (X_i - X̄)^2
Where:
n is the number of data points.
X_i is an individual data point.
X̄ is the mean (average) of the data points.
In PCA, the first principal component is the direction along which the variance of the data is maximized.
2. Covariance:
Covariance measures the degree to which two variables (dimensions or features) change together. In the context of PCA, it is used to quantify the relationships between different dimensions in the dataset. When two variables have a positive covariance, it means that they tend to increase or decrease together. When they have a negative covariance, it means that one tends to increase when the other decreases.
In mathematical terms, the covariance between two variables X and Y is calculated as:
Cov(X, Y) = (1 / (n-1)) * ∑(i=1 to n) (X_i - X̄)(Y_i - Ȳ)
Where:
n is the number of data points.
X_i and Y_i are individual data points from X and Y.
X̄ and Ȳ are the means of X and Y, respectively.
In PCA, the covariance matrix is used to determine how different dimensions of the data are related. Diagonal elements of the covariance matrix represent the variances of individual dimensions, while off-diagonal elements represent the covariances between pairs of dimensions.
The principal components in PCA are eigenvectors of the covariance matrix, and the corresponding eigenvalues indicate the amount of variance explained by each principal component. By selecting a subset of these principal components, you can reduce the dimensionality of your data while retaining as much of the original variance as possible, thus simplifying the data while preserving its essential information.
### Eigenvalues and Eigenvectors
Eigenvalues:
Eigenvalues are scalar values that indicate the amount of variance captured by each principal component (eigenvector) in a dataset. In the context of PCA, eigenvalues represent the importance or significance of each principal component. Larger eigenvalues correspond to principal components that capture more variance in the data, making them more informative.

Mathematically, for a given square matrix A, an eigenvalue (λ) is a scalar value that satisfies the equation:

�
⋅
�
=
�
⋅
�
A⋅v=λ⋅v

Where:

�
A is the square matrix.
�
v is the eigenvector associated with the eigenvalue 
�
λ.
In PCA, the eigenvalues are typically obtained by calculating the eigenvalues of the covariance matrix of the data. The eigenvalues are then sorted in descending order. The largest eigenvalues correspond to the first principal component, the second-largest to the second principal component, and so on. Eigenvalues help you determine the relative importance of each principal component and decide how many principal components to retain in your dimensionality reduction process.

Eigenvectors:
Eigenvectors are unit vectors (vectors with a magnitude of 1) associated with eigenvalues. In the context of PCA, eigenvectors represent the directions in which the data varies the most or the directions along which the variance is maximized. Each eigenvector points in a specific direction in the feature space, and these directions correspond to the principal components of the dataset.
Mathematically, for a given eigenvalue λ, the corresponding eigenvector 
v satisfies the equation:
A⋅v=λ⋅v
In PCA, the eigenvectors of the covariance matrix are computed, and each eigenvector represents a principal component. The first eigenvector corresponds to the direction of maximum variance, the second eigenvector to the second largest variance, and so on. These eigenvectors are used to create linear combinations of the original features, forming new transformed features (principal components) that capture the most essential information in the data.
The role of eigenvalues and eigenvectors in PCA can be summarized as follows:
-Eigenvalues determine the amount of variance explained by each principal component.
-Eigenvectors provide the directions (features) in which the data varies the most.
-By selecting the top k eigenvectors associated with the largest eigenvalues, you can reduce the dimensionality of your data while retaining most of the variance and preserving the essential structure of the data.
In essence, eigenvalues and eigenvectors allow PCA to identify the most important dimensions along which data varies and project the data onto these dimensions to achieve dimensionality reduction.
### Covariance Matrix
The covariance matrix is crucial to the PCA algorithm's computation of the data's main components. The pairwise covariances between the factors in the data are measured by the covariance matrix, which is a p x p matrix.
The correlation matrix C is defined as follows given a data matrix X of n observations of p variables:
C = (1/n) * X^T X
where X^T represents X's transposition. The covariances between the variables are represented by the off-diagonal elements of C, whereas the variances of the variables are represented by the diagonal elements of C.
## How PCA Works
Principal Component Analysis (PCA) is a dimensionality reduction technique used in data analysis and machine learning. It aims to reduce the complexity of high-dimensional data while preserving its essential features. Here's a step-by-step process of how PCA works:
## Step 1: Data Standardization
**Data standardization** is a crucial initial step in PCA. It involves scaling and centering the data to have a mean of 0 and a standard deviation of 1. Standardization is important because it ensures that variables with different scales contribute equally to the analysis. Without standardization, variables with larger scales may dominate the results.
## Step 2: Eigendecomposition
PCA relies on **eigendecomposition**, a linear algebra technique. It begins by calculating the covariance matrix of the standardized data. The covariance matrix represents the relationships and variances between different variables. The eigendecomposition of the covariance matrix yields eigenvectors and eigenvalues.
- **Eigenvalues** represent the amount of variance explained by each principal component. They are sorted in descending order, with the largest eigenvalue corresponding to the first principal component, the second largest to the second principal component, and so on.
- **Eigenvectors** define the direction of each principal component. These vectors are orthogonal to each other, meaning they are at right angles. The eigenvector associated with the largest eigenvalue represents the first principal component, the second largest eigenvalue represents the second principal component, and so on.
## Step 3: Selecting Principal Components
After obtaining the eigenvalues and eigenvectors, you need to decide how many principal components to retain. This is a critical decision as it determines the dimensionality of the reduced data. Common methods for selecting principal components include:
- **Explained Variance**: You can select the top N principal components that collectively explain a significant percentage of the total variance in the data. For example, you might choose to retain 95% of the variance.
- **Scree Plot**: Plotting the eigenvalues can help you identify an "elbow point" where adding more principal components does not significantly increase the explained variance. You can choose to retain the components before this point.
## Step 4: Projecting Data
The final step is to **project the original data onto the selected principal components**. This involves transforming the data into a new coordinate system defined by the principal components. The result is a reduced-dimensional representation of the data.
- The first principal component captures the most significant variation in the data.
- The second principal component captures the second most significant variation, orthogonal to the first.
- Subsequent principal components capture decreasing amounts of variation.
PCA can be applied to various domains, such as image compression, feature selection, and data visualization. It simplifies complex datasets while preserving essential information, making it a valuable tool in data analysis and machine learning.
## Implementing PCA
```python
import numpy as np

# Sample data matrix (replace this with your own data)
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Step 1: Center the data by subtracting the mean of each feature
mean = np.mean(X, axis=0)
X_centered = X - mean

# Step 2: Calculate the covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort the eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Choose the top k eigenvectors to form the new feature matrix
k = 2  # Specify the number of principal components to keep
top_k_eigenvectors = eigenvectors[:, :k]

# Step 6: Project the original data onto the new feature space
X_pca = X_centered.dot(top_k_eigenvectors)

# Now, X_pca contains the data transformed into the principal component space

# If you want to reconstruct the original data from the principal components:
X_reconstructed = X_pca.dot(top_k_eigenvectors.T) + mean
```
### Using Python and Libraries
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the Iris dataset as an example
iris = load_iris()
X = iris.data  # Features (attributes)

# Standardize the data (optional but recommended)
mean = np.mean(X, axis=0)
std_dev = np.std(X, axis=0)
X_standardized = (X - mean) / std_dev

# Create a PCA instance with the desired number of components
num_components = 2  # Specify the number of principal components you want
pca = PCA(n_components=num_components)

# Fit and transform the data
X_pca = pca.fit_transform(X_standardized)

# The principal components are in pca.components_
# The amount of variance explained by each component is in pca.explained_variance_ratio_

# Now, X_pca contains the data transformed into the principal component space
```
## Resources
Explore further resources to deepen your understanding of PCA.

### Books
1. **"Principal Component Analysis" by I. T. Jolliffe** - This is one of the seminal books on PCA, providing a thorough introduction and detailed insights into the method.

2. **"Pattern Recognition and Machine Learning" by Christopher M. Bishop** - This book covers PCA and various other machine learning topics in depth, making it a valuable resource for those interested in PCA in the context of machine learning.

3. **"Applied Multivariate Statistical Analysis" by Richard A. Johnson and Dean W. Wichern** - This book not only covers PCA but also various multivariate statistical techniques, making it suitable for readers interested in a broader perspective.

### Online Courses
1. **Coursera's "Machine Learning" by Andrew Ng** - This popular online course covers PCA and its application in machine learning. It's part of the Stanford University Machine Learning Specialization on Coursera.

2. **edX's "Principal Component Analysis (PCA) in Python"** - This course is offered by the University of California, San Diego, and provides a practical, hands-on introduction to PCA using Python.

3. **Udemy's "Dimensionality Reduction Techniques with Python"** - This course covers not only PCA but also other dimensionality reduction techniques, making it suitable for those looking to explore a broader range of methods.

### Additional Reading
1. **"A Tutorial on Principal Component Analysis" by Jonathon Shlens** - This article provides an excellent introduction to PCA, explaining the concepts and mathematics in a clear and concise manner. [Read here](https://arxiv.org/abs/1404.1100)

2. **"PCA: A Step-by-Step Example" by Sebastian Raschka** - This blog post walks through a step-by-step example of PCA with Python code. [Read here](https://sebastianraschka.com/Articles/2014_pca_step_by_step.html)

3. **"In Depth: Principal Component Analysis" by Jake VanderPlas** - This is part of the Python Data Science Handbook and provides a detailed explanation of PCA with code examples in Python. [Read here](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)

These resources should help you get a comprehensive understanding of PCA, whether you prefer books, online courses, or additional articles and papers.
