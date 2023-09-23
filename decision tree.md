## Introduction
Decision Tree Classifier is a versatile and widely used machine learning algorithm for classification tasks. In this guide, we will explore Decision Tree Classifier from fundamental concepts to advanced techniques.

## Table of Contents
1. [What is Decision Tree Classifier?](#what-is-decision-tree-classifier)
2. [How Decision Tree Works](#how-decision-tree-works)
3. [Decision Tree for Classification](#decision-tree-for-classification)
6. [Implementing Decision Tree Classifier](#implementing-decision-tree-classifier)
 8. [Resources](#resources)
   - [Books](#books)
   - [Online Courses](#online-courses)
   - [Additional Reading](#additional-reading)

## What is Decision Tree Classifier?
A Decision Tree Classifier is a tree-like structure that recursively divides the dataset into subsets based on the most significant attribute at each node. It is a powerful tool for classification tasks due to its interpretability and ease of use.
Decision trees are a popular machine learning algorithm that can be used for both regression and classification tasks. They are easy to understand, interpret, and implement, making them an ideal choice for beginners in the field of machine learning. In this comprehensive guide, we will cover all aspects of the decision tree algorithm, including the working principles, different types of decision trees, the process of building decision trees, and how to evaluate and optimize decision trees.

## How Decision Tree Works
-Root Node: The initial node at the beginning of a decision tree, where the entire population or dataset starts dividing based on various features or conditions.
-Decision Nodes: Nodes resulting from the splitting of root nodes are known as decision nodes. These nodes represent intermediate decisions or conditions within the tree.
-Leaf Nodes: Nodes where further splitting is not possible, often indicating the final classification or outcome. Leaf nodes are also referred to as terminal nodes.
-Sub-Tree: Similar to a subsection of a graph being called a sub-graph, a sub-section of a decision tree is referred to as a sub-tree. It represents a specific portion of the decision tree.
-Pruning: The process of removing or cutting down specific nodes in a decision tree to prevent overfitting and simplify the model.
-Branch / Sub-Tree: A subsection of the entire decision tree is referred to as a branch or sub-tree. It represents a specific path of decisions and outcomes within the tree.
-Parent and Child Node: In a decision tree, a node that is divided into sub-nodes is known as a parent node, and the sub-nodes emerging from it are referred to as child nodes. The parent node represents a decision or condition, while the child nodes represent the potential outcomes or further decisions based on that condition.
![498772](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/98b41e85-99f9-4667-ac35-95eca0ae606d)
Example of Decision Tree
Let’s understand decision trees with the help of an example:
![905753](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/0ce2ad48-fcd7-4a14-93f3-02e8d798a634)
Decision trees are upside down which means the root is at the top and then this root is split into various several nodes. Decision trees are nothing but a bunch of if-else statements in layman terms. It checks if the condition is true and if it is then it goes to the next node attached to that decision.

In the below diagram the tree will first ask what is the weather? Is it sunny, cloudy, or rainy? If yes then it will go to the next feature which is humidity and wind. It will again check if there is a strong wind or weak, if it’s a weak wind and it’s rainy then the person may go and play.

![542834](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/f938e29d-c67c-4279-ab33-e375aa38bf05)

Did you notice anything in the above flowchart? We see that if the weather is cloudy then we must go to play. Why didn’t it split more? Why did it stop there?

To answer this question, we need to know about few more concepts like entropy, information gain, and Gini index. But in simple terms, I can say here that the output for the training dataset is always yes for cloudy weather, since there is no disorderliness here we don’t need to split the node further.

The goal of machine learning is to decrease uncertainty or disorders from the dataset and for this, we use decision trees.

Now you must be thinking how do I know what should be the root node? what should be the decision node? when should I stop splitting? To decide this, there is a metric called “Entropy” which is the amount of uncertainty in the dataset.

### Tree Structure
Several assumptions are made to build effective models when creating decision trees. These assumptions help guide the tree’s construction and impact its performance. Here are some common assumptions and considerations when creating decision trees:

Binary Splits
Decision trees typically make binary splits, meaning each node divides the data into two subsets based on a single feature or condition. This assumes that each decision can be represented as a binary choice.

Recursive Partitioning
Decision trees use a recursive partitioning process, where each node is divided into child nodes, and this process continues until a stopping criterion is met. This assumes that data can be effectively subdivided into smaller, more manageable subsets.

Feature Independence
Decision trees often assume that the features used for splitting nodes are independent. In practice, feature independence may not hold, but decision trees can still perform well if features are correlated.

Homogeneity
Decision trees aim to create homogeneous subgroups in each node, meaning that the samples within a node are as similar as possible regarding the target variable. This assumption helps in achieving clear decision boundaries.

Top-Down Greedy Approach
Decision trees are constructed using a top-down, greedy approach, where each split is chosen to maximize information gain or minimize impurity at the current node. This may not always result in the globally optimal tree.

Categorical and Numerical Features
Decision trees can handle both categorical and numerical features. However, they may require different splitting strategies for each type.

Overfitting
Decision trees are prone to overfitting when they capture noise in the data. Pruning and setting appropriate stopping criteria are used to address this assumption.

Impurity Measures
Decision trees use impurity measures such as Gini impurity or entropy to evaluate how well a split separates classes. The choice of impurity measure can impact tree construction.

No Missing Values
Decision trees assume that there are no missing values in the dataset or that missing values have been appropriately handled through imputation or other methods.

Equal Importance of Features
Decision trees may assume equal importance for all features unless feature scaling or weighting is applied to emphasize certain features.

No Outliers
Decision trees are sensitive to outliers, and extreme values can influence their construction. Preprocessing or robust methods may be needed to handle outliers effectively.

Sensitivity to Sample Size
Small datasets may lead to overfitting, and large datasets may result in overly complex trees. The sample size and tree depth should be balanced.

Entropy
Entropy is nothing but the uncertainty in our dataset or measure of disorder. Let me try to explain this with the help of an example.

Suppose you have a group of friends who decides which movie they can watch together on Sunday. There are 2 choices for movies, one is “Lucy” and the second is “Titanic” and now everyone has to tell their choice. After everyone gives their answer we see that “Lucy” gets 4 votes and “Titanic” gets 5 votes. Which movie do we watch now? Isn’t it hard to choose 1 movie now because the votes for both the movies are somewhat equal.

This is exactly what we call disorderness, there is an equal number of votes for both the movies, and we can’t really decide which movie we should watch. It would have been much easier if the votes for “Lucy” were 8 and for “Titanic” it was 2. Here we could easily say that the majority of votes are for “Lucy” hence everyone will be watching this movie.

In a decision tree, the output is mostly “yes” or “no”
The formula for Entropy is shown below:
![706025](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/c727c00a-3fb4-462b-ab5f-db20aba242ab)
Here,

-p+ is the probability of positive class
-p– is the probability of negative class
-S is the subset of the training example

## Implementing Decision Tree Classifier
```python
import numpy as np

# Sample Iris dataset (features and labels)
# Sepal length, Sepal width, Petal length, Petal width, Label
dataset = np.array([
    [5.1, 3.5, 1.4, 0.2, 'setosa'],
    [4.9, 3.0, 1.4, 0.2, 'setosa'],
    [6.7, 3.0, 5.2, 2.3, 'virginica'],
    [6.3, 2.9, 5.6, 1.8, 'virginica'],
    # Add more data points here
])

# Define a class to represent a Decision Tree node
class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = {}  # Store child nodes
        self.label = None   # Class label (for leaf nodes)

# Calculate Gini impurity for a dataset
def gini_impurity(data):
    labels = [row[-1] for row in data]
    unique_labels = set(labels)
    impurity = 1.0
    for label in unique_labels:
        p = labels.count(label) / len(labels)
        impurity -= p ** 2
    return impurity

# Split a dataset based on a given feature and threshold
def split_dataset(data, feature_index, threshold):
    left, right = [], []
    for row in data:
        if row[feature_index] <= threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Select the best split for a dataset
def find_best_split(data):
    num_features = len(data[0]) - 1
    best_split = None
    best_gini = 1.0  # Initialize with maximum impurity
    for feature_index in range(num_features):
        values = set(row[feature_index] for row in data)
        for value in values:
            left, right = split_dataset(data, feature_index, value)
            gini = (len(left) / len(data)) * gini_impurity(left) + (len(right) / len(data)) * gini_impurity(right)
            if gini < best_gini:
                best_split = (feature_index, value)
                best_gini = gini
    return best_split

# Create a Decision Tree recursively
def create_decision_tree(data):
    labels = [row[-1] for row in data]
    # If all labels are the same, return a leaf node
    if labels.count(labels[0]) == len(labels):
        return TreeNode(data[0][-1])
    # If there are no features left to split, return the most common label
    if len(data[0]) == 1:
        return TreeNode(max(set(labels), key=labels.count))
    # Find the best split and create child nodes
    best_split = find_best_split(data)
    left_data, right_data = split_dataset(data, best_split[0], best_split[1])
    root = TreeNode(f"Feature {best_split[0]} <= {best_split[1]}")
    root.children['left'] = create_decision_tree(left_data)
    root.children['right'] = create_decision_tree(right_data)
    return root

# Predict the class label for a single data point
def predict(node, data_point):
    if node.label is not None:
        return node.label
    feature, threshold = node.data.split(' <= ')
    feature_index = int(feature.split(' ')[-1])
    if data_point[feature_index] <= float(threshold):
        return predict(node.children['left'], data_point)
    else:
        return predict(node.children['right'], data_point)

# Create the Decision Tree and make predictions
decision_tree = create_decision_tree(dataset)
sample_data_point = [5.0, 3.5, 1.5, 0.2]  # Example data point
predicted_label = predict(decision_tree, sample_data_point)
print(f"Predicted Label: {predicted_label}")
```

### Using Python and Libraries
```python
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into a training set and a testing set (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
decision_tree = DecisionTreeClassifier()

# Train the classifier on the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = decision_tree.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)
```


## Resources
Explore further resources to deepen your understanding of Decision Tree Classifier.

## Books

1. **"Introduction to Data Mining"** by Tan, Steinbach, and Kumar - This book covers Decision Trees and other data mining concepts in-depth.

2. **"Machine Learning"** by Tom Mitchell - A comprehensive textbook that includes a section on Decision Trees.

## Online Courses

3. **Coursera - "Machine Learning" by Andrew Ng** - This foundational machine learning course includes a section on Decision Trees, offering insights into theory and practical implementation.

4. **edX - "Machine Learning Fundamentals"** - A course that covers Decision Trees and other machine learning concepts suitable for beginners.

## Additional Reading

5. **[Decision Trees](https://en.wikipedia.org/wiki/Decision_tree)** - The Wikipedia page provides a detailed overview of Decision Trees, including algorithms and variants.

6. **[Understanding Decision Trees](https://towardsdatascience.com/understanding-decision-trees-6d587455ac6b)** - A comprehensive article explaining Decision Trees, their construction, and pruning.

7. **[ID3 Algorithm](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)** - Read the original paper on the ID3 (Iterative Dichotomiser 3) algorithm, which is one of the foundational Decision Tree algorithms.

8. **[CART: Classification and Regression Trees](https://www.amazon.com/CART-Classification-Regression-Trees-Breiman/dp/0412048418)** - A classic book by Leo Breiman, one of the creators of the CART algorithm, provides insights into Decision Trees for classification and regression.

## GitHub Repositories

9. **[Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/tree.html)** - The official documentation of scikit-learn provides detailed guides and examples for Decision Tree classifiers in Python.

10. **[Decision Trees in Python](https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb)** - A GitHub repository containing a Jupyter notebook with a detailed explanation and Python code for Decision Trees.

These resources cover a wide range of topics related to Decision Tree classifiers, from theory and implementation to practical applications. Whether you're a beginner or an experienced practitioner, you'll find valuable materials to deepen your understanding of this essential machine learning algorithm.
This comprehensive guide will equip you with the knowledge and skills to master Decision Tree Classifier, from the basics to advanced applications in classification and regression tasks.
