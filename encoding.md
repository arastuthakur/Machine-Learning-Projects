# Encoding Techniques in Machine Learning

## Introduction
Encoding is a crucial step in preprocessing data for machine learning models, especially when dealing with categorical variables. In this guide, we will explore various encoding techniques, from basic concepts to advanced strategies.

## Table of Contents
1. [Why Encoding Matters](#why-encoding-matters)
2. [Types of Data in Machine Learning](#types-of-data-in-machine-learning)
3. [Basic Encoding Techniques](#basic-encoding-techniques)
   - [Label Encoding](#label-encoding)
   - [One-Hot Encoding](#one-hot-encoding)
   - [Ordinal Encoding](#ordinal-encoding)
4. [Advanced Encoding Techniques](#advanced-encoding-techniques)
   - [Binary Encoding](#binary-encoding)
   - [Target Encoding](#target-encoding)
6. [Implementing Encoding Techniques](#implementing-encoding-techniques)
   - [Using Python and Libraries](#using-python-and-libraries)
   - [Best Practices](#best-practices)
7. [Resources](#resources)
   - [Books](#books)
   - [Online Courses](#online-courses)
   - [Additional Reading](#additional-reading)

## Why Encoding Matters
Machine learning models can only work with numerical values. For this reason, it is necessary to transform the categorical values of the relevant features into numerical ones. This process is called feature encoding.
Data frame analytics automatically performs feature encoding. The input data is pre-processed with the following encoding techniques:
one-hot encoding: Assigns vectors to each category. The vector represent whether the corresponding feature is present (1) or not (0).
target-mean encoding: Replaces categorical values with the mean value of the target variable.
frequency encoding: Takes into account how many times a given categorical value is present in relation with a feature.
When the model makes predictions on new data, the data needs to be processed in the same way it was trained. Machine learning model inference in the Elastic Stack does this automatically, so the automatically applied encodings are used in each call for inference. Refer to inference for classification and regression.
Feature importance is calculated for the original categorical fields, not the automatically encoded features.
## Types of Data in Machine Learning
Numerical Data
Numerical data is any data where data points are exact numbers. Statisticians also might call numerical data, quantitative data. This data has meaning as a measurement such as house prices or as a count, such as a number of residential properties in Los Angeles or how many houses sold in the past year.

Numerical data can be characterized by continuous or discrete data. Continuous data can assume any value within a range whereas discrete data has distinct values.


Numerical Data
For example, the number of students taking Python class would be a discrete data set. You can only have discrete whole number values like 10, 25, or 33. A class cannot have 12.75 students enrolled. A student either join a class or he doesn’t. On the other hand, continuous data are numbers that can fall anywhere within a range. Like a student could have an average score of 88.25 which falls between 0 and 100.

The takeaway here is that numerical data is not ordered in time. They are just numbers that we have collected.

Categorical Data
Categorical data represents characteristics, such as a hockey player’s position, team, hometown. Categorical data can take numerical values. For example, maybe we would use 1 for the colour red and 2 for blue. But these numbers don’t have a mathematical meaning. That is, we can’t add them together or take the average.

In the context of super classification, categorical data would be the class label. This would also be something like if a person is a man or woman, or property is residential or commercial.

There is also something called ordinal data, which in some sense is a mix of numerical and categorical data. In ordinal data, the data still falls into categories, but those categories are ordered or ranked in some particular way. An example would be class difficulty, such as beginner, intermediate, and advanced. Those three types of classes would be a way that we could label the classes, and they have a natural order in increasing difficulty.


Another example is that we just take quantitative data, and splitting it into groups, so we have bins or categories of other types of data.


Ordinal Data
For plotting purposes, ordinal data is treated much in the same way as categorical data. But groups are usually ordered from lowest to highest so that we can preserve this ordering.

Time Series Data
Time series data is a sequence of numbers collected at regular intervals over some period of time. It is very important, especially in particular fields like finance. Time series data has a temporal value attached to it, so this would be something like a date or a timestamp that you can look for trends in time.

For example, we might measure the average number of home sales for many years. The difference of time series data and numerical data is that rather than having a bunch of numerical values that don’t have any time ordering, time-series data does have some implied ordering. There is a first data point collected and the last data point collected.


CREA
Text
Text data is basically just words. A lot of the time the first thing that you do with text is you turn it into numbers using some interesting functions like the bag of words formulation.
These are four types of data from a Machine Learning perspective. Depending on exactly the type of data, this might have some repercussions for the type of algorithms that you can use for feature engineering and modeling, or the type of questions that you can ask of it.

## Basic Encoding Techniques
Learn fundamental encoding techniques for categorical variables.
### Label Encoding
Label encoding is a technique used in machine learning and data analysis to convert categorical variables into numerical format. It is particularly useful when working with algorithms that require numerical input, as most machine learning models can only operate on numerical data. In this explanation, we’ll explore how label encoding works and how to implement it in Python.

Let’s consider a simple example with a dataset containing information about different types of fruits, where the “Fruit” column has categorical values such as “Apple,” “Orange,” and “Banana.” Label encoding assigns a unique numerical label to each distinct category, transforming the categorical data into numerical representation.

### One-Hot Encoding
For categorical variables where no such ordinal relationship exists, the integer encoding is not enough.

In fact, using this encoding and allowing the model to assume a natural ordering between categories may result in poor performance or unexpected results (predictions halfway between categories).

In this case, a one-hot encoding can be applied to the integer representation. This is where the integer encoded variable is removed and a new binary variable is added for each unique integer value.

In the “color” variable example, there are 3 categories and therefore 3 binary variables are needed. A “1” value is placed in the binary variable for the color and “0” values for the other colors.

### Ordinal Encoding
Ordinal encoding is a technique to transform categorical features into a numerical format. In ordinal encoding, labels are translated to numbers based on their ordinal relationship to one another. For example, if one feature contains - {low, medium, high}, it can be converted into {1,2,3}, where 1 represents low, 2 represents medium, and 3 represents high. It is one of the essential tasks before training an ML model, as many ML algorithms do not support categorical data directly and require them to be converted into a numerical format.

## Advanced Encoding Techniques
Explore advanced encoding methods for improving model performance.

### Binary Encoding
Binary encoding is a combination of Hash encoding and one-hot encoding. In this encoding scheme, the categorical feature is first converted into numerical using an ordinal encoder. Then the numbers are transformed in the binary number. After that binary value is split into different columns.

Binary encoding works really well when there are a high number of categories. For example the cities in a country where a company supplies its products.

### Target Encoding
Target encoding is a technique used in machine learning and data preprocessing to transform categorical variables into numerical values. Unlike one-hot encoding, which creates binary columns for each category, target encoding calculates and assigns a numerical value to each category based on the relationship between the category and the target variable. Typically used for classification tasks, it replaces the categorical values with their corresponding mean (or other statistical measures) of the target variable within each category.

Target encoding can be effective in capturing valuable information from categorical data while reducing the dimensionality of the feature space, making it suitable for models like decision trees and gradient boosting.

Target encoding is a Baysian encoding technique.

## Implementing Encoding Techniques
Learn how to implement encoding techniques effectively.

### Using Python and Libraries
```python
from sklearn.preprocessing import OneHotEncoder

# Sample categorical data
data = ["cat", "dog", "fish", "dog", "cat"]

# Create a one-hot encoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the data
encoded_data = encoder.fit_transform(np.array(data).reshape(-1, 1))

# Display the encoded data
print(encoded_data)
from sklearn.preprocessing import LabelEncoder

# Sample categorical data
data = ["low", "medium", "high", "medium", "low"]

# Create a label encoder
encoder = LabelEncoder()

# Fit and transform the data
encoded_data = encoder.fit_transform(data)

# Display the encoded data
print(encoded_data)
```
## Resources
Explore further resources to deepen your understanding of encoding techniques in machine learning.

## Books

1. **"Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists"** by Alice Zheng and Amanda Casari - This book covers encoding techniques as part of the feature engineering process.

2. **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron - A comprehensive book that includes a section on encoding categorical data with practical examples.

## Online Courses

3. **Coursera - "Feature Engineering for Machine Learning"** - A course that dives deep into feature engineering techniques, including encoding, taught by leading experts in the field.

4. **edX - "Machine Learning Fundamentals"** - An online course that covers encoding techniques and other essential concepts for machine learning.

## Additional Reading

5. **[Categorical Data Encoding: Introduction and Basics](https://towardsdatascience.com/categorical-data-encoding-introduction-and-basics-6a57cadb3dfe)** - An article that provides an introduction to categorical data encoding techniques.

6. **[The Cardinal Sin of Categorical Data](https://towardsdatascience.com/the-cardinal-sin-of-categorical-data-its-not-about-cards-any-more-360e6c59d05f)** - A detailed article discussing the challenges of categorical data and encoding strategies.

7. **[Encoding Categorical Data](https://towardsdatascience.com/encoding-categorical-data-21a2651a065c)** - An in-depth article covering various encoding techniques and their applications.

8. **[Categorical Data Encoding Techniques](https://medium.com/data-design/categorical-data-encoding-techniques-a0e107df978d)** - A Medium article exploring different encoding techniques with code examples.

## GitHub Repositories

9. **[Categorical Encoding Methods (GitHub)](https://github.com/scikit-learn-contrib/categorical-encoding)** - A GitHub repository containing a collection of categorical encoding methods for scikit-learn.

10. **[Feature Engineering Recipes (GitHub)](https://github.com/DiploDatos/FeatureEngineering)** - A repository that includes Jupyter notebooks on feature engineering and data preprocessing, including encoding.

These resources cover a wide range of topics related to encoding techniques in machine learning, from books and courses to articles, code examples, and practical tips. Whether you're looking to enhance your encoding skills or start from scratch, these materials can be valuable in your journey.

This comprehensive guide will equip you with the knowledge and skills to effectively encode categorical data in machine learning projects, ultimately improving model accuracy and robustness.
