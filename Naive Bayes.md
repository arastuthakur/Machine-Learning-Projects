# Naive Bayes Explained

## Introduction
Naive Bayes is a simple yet powerful probabilistic machine learning algorithm used for classification and text analysis tasks. In this guide, we will explore Naive Bayes from basic concepts to advanced techniques.

## Table of Contents
1. [What is Naive Bayes?](#what-is-naive-bayes)
2. [Bayes' Theorem](#bayes-theorem)
3. [Types of Naive Bayes Classifiers](#types-of-naive-bayes-classifiers)
   - [Multinomial Naive Bayes](#multinomial-naive-bayes)
   - [Gaussian Naive Bayes](#gaussian-naive-bayes)
   - [Bernoulli Naive Bayes](#bernoulli-naive-bayes)
4. [Training a Naive Bayes Classifier](#training-a-naive-bayes-classifier)
5. [Implementing Naive Bayes](#implementing-naive-bayes)
   - [Using Python and Libraries](#using-python-and-libraries)   
6. [Resources](#resources)
   - [Books](#books)
   - [Online Courses](#online-courses)
   - [Additional Reading](#additional-reading)

## What is Naive Bayes?
Naive Bayes is a probabilistic algorithm based on Bayes' theorem that makes strong independence assumptions between features. It is widely used for classification problems, especially in natural language processing and spam detection.
In statistics, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features (see Bayes classifier). They are among the simplest Bayesian network models,[1] but coupled with kernel density estimation, they can achieve high accuracy levels.[2]

Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem. Maximum-likelihood training can be done by evaluating a closed-form expression, which takes linear time, rather than by expensive iterative approximation as used for many other types of classifiers.

In the statistics literature, naive Bayes models are known under a variety of names, including simple Bayes and independence Bayes. All these names reference the use of Bayes' theorem in the classifier's decision rule, but naive Bayes is not (necessarily) a Bayesian method.
## Bayes' Theorem
Bayes' Theorem thus gives the probability of an event based on new information that is or may be related to that event. The formula also can be used to determine how the probability of an event occurring may be affected by hypothetical new information, supposing the new information will turn out to be true.

For instance, consider drawing a single card from a complete deck of 52 cards.

There are four kings in the deck, so the probability that the card is a king is four divided by 52, which equals 1/13 or approximately 7.69%. Now, suppose it is revealed that the selected card is a face card. The probability the selected card is a king, given it is a face card, is four divided by 12, or approximately 33.3%, as there are 12 face cards in a deck.
<img width="293" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/f25afd43-3c59-4e9f-a4f2-63cb9d727aeb">

## Types of Naive Bayes Classifiers
Different types of Naive Bayes classifiers are tailored for different types of data.

### Multinomial Naive Bayes
Multinomial Naïve Bayes consider a feature vector where a given term represents the number of times it appears or very often i.e. frequency. On the other hand, Bernoulli is a binary algorithm used when the feature is present or not. At last Gaussian is based on continuous distribution.

Advantages:
-Low computation cost.
-It can effectively work with large datasets.
-For small sample sizes, Naive Bayes can outperform the most powerful alternatives.
-Easy to implement, fast and accurate method of prediction.
-Can work with multiclass prediction problems.
-It performs well in text classification problems.
Disadvantages:
-It is very difficult to get the set of independent predictors for developing a model using Naive Bayes.

Applications:
-Naive Bayes classifier is used in Text Classification, Spam filtering and Sentiment Analysis. It has a higher success rate than other algorithms.
-Naïve Bayes along with Collaborative filtering are used in Recommended Systems.
-It is also used in disease prediction based on health parameters.
-This algorithm has also found its application in Face recognition.
-Naive Bayes is used in prediction of weather reports based on atmospheric conditions (temp, wind, clouds, humidity etc.)

### Bernoulli Naive Bayes
Before going ahead, let us have a look at the Bernoulli Distribution:-

 

Let there be a random variable 'X' and let the probability of success be denoted by 'p' and the likelihood of failure be represented by 'q.'  

 

Success: p 

Failure: q 

 

q = 1 - (probability of Sucesss)

q = 1 - p

 


 

As we notice above, x can take only two values (binary values), i.e., 0 or 1.  

 

Bernoulli Naive Bayes is a part of the Naive Bayes family. It is based on the Bernoulli Distribution and accepts only binary values, i.e., 0 or 1. If the features of the dataset are binary, then we can assume that Bernoulli Naive Bayes is the algorithm to be used. 
<img width="464" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/9a16aff7-86da-4845-8b01-1bc27d3cffd8">

## Training a Naive Bayes Classifier
First Approach (In case of a single feature)
Naive Bayes classifier calculates the probability of an event in the following steps:

Step 1: Calculate the prior probability for given class labels
Step 2: Find Likelihood probability with each attribute for each class
Step 3: Put these value in Bayes Formula and calculate posterior probability.
Step 4: See which class has a higher probability, given the input belongs to the higher probability class.
For simplifying prior and posterior probability calculation, you can use the two tables frequency and likelihood tables. Both of these tables will help you to calculate the prior and posterior probability. The Frequency table contains the occurrence of labels for all features. There are two likelihood tables. Likelihood Table 1 is showing prior probabilities of labels and Likelihood Table 2 is showing the posterior probability.

<img width="523" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/439f7222-4a12-4a31-96e4-8a649385bb3a">

Now suppose you want to calculate the probability of playing when the weather is overcast.

Probability of playing:

P(Yes | Overcast) = P(Overcast | Yes) P(Yes) / P (Overcast) .....................(1)

Calculate Prior Probabilities:

P(Overcast) = 4/14 = 0.29

P(Yes)= 9/14 = 0.64

 
Calculate Posterior Probabilities:

P(Overcast |Yes) = 4/9 = 0.44

 
Put Prior and Posterior probabilities in equation (1)

P (Yes | Overcast) = 0.44 * 0.64 / 0.29 = 0.98(Higher)

Similarly, you can calculate the probability of not playing:

Probability of not playing:

P(No | Overcast) = P(Overcast | No) P(No) / P (Overcast) .....................(2)

Calculate Prior Probabilities:

P(Overcast) = 4/14 = 0.29

P(No)= 5/14 = 0.36

 
Calculate Posterior Probabilities:

P(Overcast |No) = 0/9 = 0

 
Put Prior and Posterior probabilities in equation (2)

P (No | Overcast) = 0 * 0.36 / 0.29 = 0

The probability of a 'Yes' class is higher. So you can determine here if the weather is overcast than players will play the sport.

Second Approach (In case of multiple features)
How Naive Bayes classifier works?
Now suppose you want to calculate the probability of playing when the weather is overcast, and the temperature is mild.

Probability of playing:

P(Play= Yes | Weather=Overcast, Temp=Mild) = P(Weather=Overcast, Temp=Mild | Play= Yes)P(Play=Yes) ..........(1)

P(Weather=Overcast, Temp=Mild | Play= Yes)= P(Overcast |Yes) P(Mild |Yes) ………..(2)

Calculate Prior Probabilities: P(Yes)= 9/14 = 0.64

Calculate Posterior Probabilities: P(Overcast |Yes) = 4/9 = 0.44 P(Mild |Yes) = 4/9 = 0.44

Put Posterior probabilities in equation (2) P(Weather=Overcast, Temp=Mild | Play= Yes) = 0.44 * 0.44 = 0.1936(Higher)

Put Prior and Posterior probabilities in equation (1) P(Play= Yes | Weather=Overcast, Temp=Mild) = 0.1936*0.64 = 0.124

Similarly, you can calculate the probability of not playing:

Probability of not playing:

P(Play= No | Weather=Overcast, Temp=Mild) = P(Weather=Overcast, Temp=Mild | Play= No)P(Play=No) ..........(3)

P(Weather=Overcast, Temp=Mild | Play= No)= P(Weather=Overcast |Play=No) P(Temp=Mild | Play=No) ………..(4)

Calculate Prior Probabilities: P(No)= 5/14 = 0.36

Calculate Posterior Probabilities: P(Weather=Overcast |Play=No) = 0/9 = 0 P(Temp=Mild | Play=No)=2/5=0.4

Put posterior probabilities in equation (4) P(Weather=Overcast, Temp=Mild | Play= No) = 0 * 0.4= 0

Put prior and posterior probabilities in equation (3) P(Play= No | Weather=Overcast, Temp=Mild) = 0*0.36=0

The probability of a 'Yes' class is higher. So you can say here that if the weather is overcast than players will play the sport.


### Pros and Cons
Pros:

It is easy and fast to predict class of test data set. It also perform well in multi class prediction
When assumption of independence holds, the classifier performs better compared to other machine learning models like logistic regression or decision tree, and requires less training data.
It perform well in case of categorical input variables compared to numerical variable(s). For numerical variable, normal distribution is assumed (bell curve, which is a strong assumption).
Cons:

If categorical variable has a category (in test data set), which was not observed in training data set, then model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as “Zero Frequency”. To solve this, we can use the smoothing technique. One of the simplest smoothing techniques is called Laplace estimation.
On the other side, Naive Bayes is also known as a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.
Another limitation of this algorithm is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.


## Implementing Naive Bayes
Learn how to implement Naive Bayes models.
```Python
import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}  # P(spam) and P(non-spam)
        self.word_probs = {'spam': {}, 'non-spam': {}}

    def fit(self, X_train, y_train):
        # Calculate class probabilities
        total_samples = len(y_train)
        self.class_probs['spam'] = sum(1 for label in y_train if label == 'spam') / total_samples
        self.class_probs['non-spam'] = 1 - self.class_probs['spam']

        # Calculate word probabilities
        for label in ['spam', 'non-spam']:
            label_indices = np.where(y_train == label)[0]
            label_documents = [X_train[i] for i in label_indices]

            # Flatten the list of documents and count word occurrences
            words = ' '.join(label_documents).split()
            word_counts = {}
            total_words = len(words)

            for word in words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1

            # Calculate P(word | class)
            for word, count in word_counts.items():
                self.word_probs[label][word] = count / total_words

    def predict(self, X_test):
        predictions = []

        for document in X_test:
            # Calculate posterior probabilities for each class
            probs = {'spam': np.log(self.class_probs['spam']), 'non-spam': np.log(self.class_probs['non-spam'])}

            for label in ['spam', 'non-spam']:
                for word in document.split():
                    if word in self.word_probs[label]:
                        probs[label] += np.log(self.word_probs[label][word])

            # Choose the class with the highest posterior probability
            prediction = max(probs, key=probs.get)
            predictions.append(prediction)

        return predictions
# Example usage
X_train = ["spam email text", "non-spam email text", ...]  # Training text data
y_train = ["spam", "non-spam", ...]  # Corresponding labels

X_test = ["new email text", ...]  # Test text data

# Initialize and train the classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# Make predictions
predictions = nb_classifier.predict(X_test)

```

### Using Python and Libraries
```Python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X = newsgroups.data
y = newsgroups.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using a bag-of-words representation
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Initialize and train the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=newsgroups.target_names)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)
```


## Resources
Explore further resources to deepen your understanding of Naive Bayes.

## Books

1. **"Pattern Classification"** by Richard O. Duda, Peter E. Hart, and David G. Stork - This book covers Naive Bayes as part of the broader topic of pattern classification.

2. **"Machine Learning: A Probabilistic Perspective"** by Kevin P. Murphy - A comprehensive resource that includes a detailed chapter on Naive Bayes and its probabilistic foundations.

## Online Courses

3. **Coursera - "Machine Learning" by Andrew Ng** - This foundational machine learning course covers Naive Bayes as part of its curriculum, providing a clear understanding of the algorithm's principles.

4. **edX - "Introduction to Artificial Intelligence"** - A course that introduces Naive Bayes and other machine learning techniques, suitable for beginners.

## Additional Reading

5. **[A Naive Bayesian Classifier for Spam Detection](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)** - A classic paper on using Naive Bayes for spam detection.

6. **[Naive Bayes and Text Classification: A Review](https://arxiv.org/abs/1410.5329)** - An academic paper that provides a comprehensive review of Naive Bayes for text classification.

7. **[The Optimality of Naive Bayes](https://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf)** - A paper discussing the optimality properties of Naive Bayes under certain conditions.

8. **[A Comparative Study of Text Classification Algorithms](https://dl.acm.org/doi/10.1145/502585.502704)** - A research paper that compares Naive Bayes with other text classification algorithms.

## GitHub Repositories

9. **[Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)** - The official documentation of scikit-learn provides detailed guides and examples for using Naive Bayes in Python.

10. **[Naive Bayes Classifier in Python](https://github.com/dipanjanS/nb_text_classifier)** - A GitHub repository containing a Python implementation of a Naive Bayes text classifier.

