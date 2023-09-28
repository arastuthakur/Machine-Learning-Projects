# Feature Engineering Explained

## Introduction
Feature Engineering is a crucial and often underestimated step in the data preprocessing pipeline of machine learning projects. In this guide, we will explore Feature Engineering from basic concepts to advanced techniques.

## Table of Contents
1. [What is Feature Engineering?](#what-is-feature-engineering)
2. [Why is Feature Engineering Important?](#why-is-feature-engineering-important)
3. [Basic Feature Engineering Techniques](#basic-feature-engineering-techniques)
   - [Handling Missing Data](#handling-missing-data)
   - [Encoding Categorical Variables](#encoding-categorical-variables)
   - [Feature Scaling](#feature-scaling)
   - [Creating Interaction Features](#creating-interaction-features)
4. [Feature Transformation](#feature-transformation)
   - [Log Transformation](#log-transformation)
   - [Box-Cox Transformation](#box-cox-transformation)
   - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
5. [Advanced Feature Engineering Techniques](#advanced-feature-engineering-techniques)
   - [Feature Selection](#feature-selection)
6. [Resources](#resources)
   - [Books](#books)
   - [Online Courses](#online-courses)
   - [Additional Reading](#additional-reading)

## What is Feature Engineering?
Feature Engineering involves creating, selecting, and transforming features (variables) in a dataset to improve the performance of machine learning models.
Feature engineering refers to the process of using domain knowledge to select and transform the most relevant variables from raw data when creating a predictive model using machine learning or statistical modeling. The goal of feature engineering and selection is to improve the performance of machine learning (ML) algorithms.
‍![6230e9ee021b250dd3710f8e_61ca4fbcc80819e696ba0ee9_Feature-Engineering-Machine-Learning-Diagram](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/7d2ba07e-3ff7-4402-afc1-e29525d47151)
Image from Practice Machine Learning with Python, Appress/Springer
## Why is Feature Engineering Important?
Feature engineering is a vital part of this. Without this step, the accuracy of your machine learning algorithm reduces significantly.
<img width="463" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/24f35258-39d3-45fe-b929-ffe75f13cc12">
A typical machine learning starts with data collection and exploratory analysis. Data cleaning comes next. This step removes duplicate values and correcting mislabelled classes and features.
Feature engineering is the next step. The output from feature engineering is fed to the predictive models, and the results are cross-validated.      
An algorithm that is fed the raw data is unaware of the importance of the features. It is making predictions in the dark.
You can think of feature engineering as the guiding light in this scenario.  
When you have relevant features, the complexity of the algorithms reduces. Even if you use an algorithm that is not ideal for the situation, the results will still be accurate.
Simpler models are often easier to understand, code, and maintain. 
The winning teams in Kaggle competitions admit to focussing more on feature engineering and data cleaning. 
The most valid answer to the question – what is feature engineering is that it is a guide to your algorithms.
You can also use your domain knowledge to engineer the features and focus on the most relevant aspects of the data. 

## Basic Feature Engineering Techniques
Learn fundamental techniques for feature engineering.
1. Imputation
   ![Multiple_imputation_62629fba75b994cafe28f8e635adbff8](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/58589621-9e46-4bf9-9a33-c98f94acd102)
The dataset is not always complete. Values may be missing from the columns. Imputation is used to fill in the missing values.
The simplest way to perform imputation would be to fill in a default value. The column may contain categorical data. You can assign the most common category. 
If the ML algorithm has to make accurate predictions, the missing training data should be filled with a value that is as close to the original as possible. You can use statistical imputation for this.  
Here, you are creating a predictive model within another predictive model. The complexity can become uncontrollable in such cases.
Therefore, the imputation method should be fast and its prediction equation should be simple. You should also ensure that the predictor for the missing data is not too sensitive to outliers. 
K-nearest neighbors, tree-based models, and linear models are some of the most common techniques used for imputation.
2. Handling Outliers
   ![780px-Diagrama_de_caixa_com_outliers_and_whisker_12123649de49471242c6f5be6e5e4b09](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/76071d6e-7cb7-430d-a24b-289da9be7dce)
Any observation that is too far from the rest is known as an outlier. It impacts the predictive models and distorts the results. The first step in handling outliers is detecting them.
You can use box plots, z-score, or Cook’s distance to identify the outliers. 
When it comes to recognising outliers, visualisation is a better approach. It gives more accurate results and makes it easier to spot the outliers.
The outliers can be removed from the dataset. But this also reduces the size of your training data. Trimming the outliers makes sense when you have a large number of observations. 
What is a feature engineering technique that can be used to solve this problem? You can also Winsorise the data or use log-scale transformation on it.
Models such as Random Forests and Gradient Boosting are resilient to the impact of outliers. If there are too many outliers, it makes better sense to switch to these tree-based models. 
3. Grouping Operation
   Every feature has different classes. Some of these classes may only have a few observations. These are called sparse classes. They can make the algorithm overfit the data.
Overfitting is a common pitfall in ML models that needs to be avoided to create a flexible model. You can group such classes to create a new one. You can start by grouping similar classes.
Grouping also works in a different context. When some features are combined, they provide more information than if they were separated. These features are called interaction features.
For example, if your dataset has the sales information of two different items in two columns and you are interested in total sales, you can add these two features. You can multiply, add, subtract, or divide two features.     
4. Feature Split
Feature splitting is the opposite of grouping or interaction features. In grouping operations, you combine two or more features to create a new one.
In feature splitting, you split a single feature into two or more parts to get the necessary information. 
For example, if the name column contains both first and last name but you are interested only in the first name, splitting the name feature into two would be a better option.  
Feature splitting is most commonly used on features that contain long strings. Splitting these make it easier for the machine learning algorithm to understand and utilize them.
It also becomes easier to perform other feature engineering techniques. Feature splitting is a vital step in improving the performance of the model.
5. Binning
Binning transforms continuous-valued features into categorical features. You can group these continuous values into a pre-defined number of bins.
Binning is used to preventing overfitting of data and make the model robust. However, binning comes at a cost. You end up losing information, and this loss can negatively impact the performance of the model. 
![Chart-graph-analytics-business-finance](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/e788317f-1d0e-45d6-be4d-878977b811f6)
You need to find a balance between overfitting and improved performance. Consider the number of views per video on Youtube.
It can be abnormally large for some videos or extremely low for some others. Using this column without binning can lead to performance issues and wrong predictions.
The bins can have fixed or adaptive width. If the data is distributed almost uniformly, then fixed-width binning is sufficient. However, when the data distribution is irregular, adaptive binning offers a better outcome. 
6. Log Transform
Is your data distributed normally? Or is it skewed? A skewed dataset leads to inadequate performance of the model.
Logarithmic transformations can fix the skewness and make the model close to normal. Logarithmic transformation is also helpful when the magnitude order of the data within the same range.
<img width="462" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/94c5fb3d-2808-4ebe-a4aa-6799660a1436">
Log transformation also reduces the effect of outliers. Outliers are a common occurrence in many datasets. If you set out to delete all outliers, you also end losing valuable information.
When your dataset is small in size, deleting outliers is not the ideal solution.
Log transformation retains the outliers but reduces the impact they have on the data. It makes the data more robust.    
Keep in mind that the log transform only works for positive values.
If your data has negative values, then you need to add a constant to the whole column to make it positive and then use this technique.     

### Handling Missing Data
Multiple approaches exist for handling missing data. This section covers some of them along with their benefits and drawbacks.
Data Dropping
Using the dropna() function is the easiest way to remove observations or features with missing values from the dataframe. Below are some techniques. 
1) Drop observations with missing values
dropna(): drops all the rows with missing values.
dropna(thresh = minimum_value): drop rows based on a threshold. This strategy sets a minimum number of missing values required to preserve the rows.
2) Drop columns with missing values
The parameter axis = 1 can be used to explicitly specify we are interested in columns rather than rows.
dropna(axis = 1): drops all the columns with missing values.
-Mean/Median Imputation
These replacement strategies  are self-explanatory. Mean and median imputations are respectively used to replace missing values of a given column with the mean and median of the non-missing values in that column. 
Normal distribution is the ideal scenario. Unfortunately, it is not always the case. This is where the median imputation can be helpful because it is not sensitive to outliers.
In Python, the fillna() function from pandas can be used to make these replacements.
-Random Sample Imputation
The idea behind the random sample imputation is different from the previous ones and involves additional steps. 
First, it starts by creating two subsets from the original data. 
The first subset contains all the observations without missing data, and the second one contains those with missing data. 
Then, it randomly selects from each subset a random observation.
Furthermore, the missing data from the previously selected observation is replaced with the existing ones from the observation having all the data available.
Finally, the process continues until there is no more missing information.
-Multiple Imputation
This is a multivariate imputation technique, meaning that the missing information is filled by taking into consideration the information from the other columns. 

For instance, if the income value is missing for an individual, it is uncertain whether or not they have a mortgage. So, to determine the correct value, it is necessary to evaluate other characteristics such as credit score, occupation, and whether or not the individual owns a house.
Multiple Imputation by Chained Equations (MICE for short) is one of the most popular imputation methods in multivariate imputation. To better understand the MICE approach, let’s consider the set of variables X1, X2, … Xn, where some or all have missing values. 
The algorithm works as follows: 
For each variable, replace the missing value with a simple imputation strategy such as mean imputation, also considered as “placeholders.”
The “placeholders” for the first variable, X1, are regressed by using a regression model where X1 is the dependent variable, and the rest of the variables are the independent variables. Then X2 is used as dependent variables and the rest as independent variables. The process continues as such until all the variables are considered at least once as the dependent variable.
Those original “placeholders” are then replaced with the predictions from the regression model.
The replacement process is repeated for a number of cycles which is generally ten, according to Raghunathan et al. 2002, and the imputation is updated at each cycle. 
At the end of the cycle, the missing values are ideally replaced with the prediction values that best reflect the relationships identified in the data.

### Encoding Categorical Variables
What is Categorical Data?
Since we are going to be working on categorical variables in this article, here is a quick refresher on the same with a couple of examples. Categorical variables are usually represented as ‘strings’ or ‘categories’ and are finite in number. Here are a few examples:
The city where a person lives: Delhi, Mumbai, Ahmedabad, Bangalore, etc.
The department a person works in: Finance, Human resources, IT, Production.
The highest degree a person has: High school, Diploma, Bachelors, Masters, PhD.
The grades of a student:  A+, A, B+, B, B- etc.
In the above examples, the variables only have definite possible values. Further, we can see there are two kinds of categorical data-
-Ordinal Data: The categories have an inherent order
-Nominal Data: The categories do not have an inherent order
In Ordinal data, while encoding, one should retain the information regarding the order in which the category is provided. Like in the above example the highest degree a person possesses, gives vital information about his qualification. The degree is an important feature to decide whether a person is suitable for a post or not.
While encoding Nominal data, we have to consider the presence or absence of a feature. In such a case, no notion of order is present. For example, the city a person lives in. For the data, it is important to retain where a person lives. Here, We do not have any order or sequence. It is equal if a person lives in Delhi or Bangalore.
For encoding categorical data, we have a python package category_encoders. The following code helps you install easily.
```
pip install category_encoders
```
Label Encoding or Ordinal Encoding
We use this categorical data encoding technique when the categorical feature is ordinal. In this case, retaining the order is important. Hence encoding should reflect the sequence.
In Label encoding, each label is converted into an integer value. We will create a variable that contains the categories representing the education qualification of a person.
```python
import category_encoders as ce
import pandas as pd
train_df=pd.DataFrame({'Degree':['High school','Masters','Diploma','Bachelors','Bachelors','Masters','Phd','High school','High school']})

# create object of Ordinalencoding
encoder= ce.OrdinalEncoder(cols=['Degree'],return_df=True,
                           mapping=[{'col':'Degree',
'mapping':{'None':0,'High school':1,'Diploma':2,'Bachelors':3,'Masters':4,'phd':5}}])

#Original data
print(train_df)
```
Fit and transform train data
df_train_transformed = encoder.fit_transform(train_df)
<img width="120" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/f4571f32-e6a7-4a18-a31c-2d3d8851ea0e">
One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.
These newly created binary features are known as Dummy variables. The number of dummy variables depends on the levels present in the categorical variable. This might sound complicated. Let us take an example to understand this better. Suppose we have a dataset with a category animal, having different animals like Dog, Cat, Sheep, Cow, Lion. Now we have to one-hot encode this data.
<img width="631" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/511bdeb1-d9ad-4a25-ac1e-44c031b47552">
After encoding, in the second table, we have dummy variables each representing a category in the feature Animal. Now for each category that is present, we have 1 in the column of that category and 0 for the others. Let’s see how to implement a one-hot encoding in python.
```python
import category_encoders as ce
import pandas as pd
data=pd.DataFrame({'City':[
'Delhi','Mumbai','Hydrabad','Chennai','Bangalore','Delhi','Hydrabad','Bangalore','Delhi'
]})

#Create object for one-hot encoding
encoder=ce.OneHotEncoder(cols='City',handle_unknown='return_nan',return_df=True,use_cat_names=True)

#Original Data
data
```
<img width="167" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/8641ac88-7d5d-4e98-b249-46312a1e911e">
```python
#Fit and transform Data
data_encoded = encoder.fit_transform(data)
data_encoded
```
<img width="512" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/6f01121f-9d1a-47fa-96ed-9929bd41c47b">
Now let’s move to another very interesting and widely used encoding technique i.e Dummy encoding.
Dummy Encoding
Dummy coding scheme is similar to one-hot encoding. This categorical data encoding method transforms the categorical variable into a set of binary variables (also known as dummy variables). In the case of one-hot encoding, for N categories in a variable, it uses N binary variables. The dummy encoding is a small improvement over one-hot-encoding. Dummy encoding uses N-1 features to represent N labels/categories.
To understand this better let’s see the image below. Here we are coding the same data using both one-hot encoding and dummy encoding techniques. While one-hot uses 3 variables to represent the data whereas dummy encoding uses 2 variables to code 3 categories.
<img width="471" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/24b4483f-d773-4550-9059-92c8b61a25f1">
Let us implement it in python.
```python
import category_encoders as ce
import pandas as pd
data=pd.DataFrame({'City':['Delhi','Mumbai','Hyderabad','Chennai','Bangalore','Delhi,'Hyderabad']})

#Original Data
data
``
<img width="128" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/f9a4b3b8-e50b-4632-a785-cb618e7e6208">
```python
#encode the data
data_encoded=pd.get_dummies(data=data,drop_first=True)
data_encoded
```
<img width="382" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/589f907a-b654-472f-ab39-98f78951a693">
Here using drop_first  argument, we are representing the first label Bangalore using 0.
Drawbacks of One-Hot and Dummy Encoding
One hot encoder and dummy encoder are two powerful and effective encoding schemes. They are also very popular among the data scientists, But may not be as effective when-
A large number of levels are present in data. If there are multiple categories in a feature variable in such a case we need a similar number of dummy variables to encode the data. For example, a column with 30 different values will require 30 new variables for coding.
If we have multiple categorical features in the dataset similar situation will occur and again we will end to have several binary features each representing the categorical feature and their multiple categories e.g a dataset having 10 or more categorical columns.
In both the above cases, these two encoding schemes introduce sparsity in the dataset i.e several columns having 0s and a few of them having 1s. In other words, it creates multiple dummy features in the dataset without adding much information.
Also, they might lead to a Dummy variable trap. It is a phenomenon where features are highly correlated. That means using the other variables, we can easily predict the value of a variable.
Due to the massive increase in the dataset, coding slows down the learning of the model along with deteriorating the overall performance that ultimately makes the model computationally expensive. Further, while using tree-based models these encodings are not an optimum choice.
Effect Encoding
This encoding technique is also known as Deviation Encoding or Sum Encoding. Effect encoding is almost similar to dummy encoding, with a little difference. In dummy coding, we use 0 and 1 to represent the data but in effect encoding, we use three values i.e. 1,0, and -1.
The row containing only 0s in dummy encoding is encoded as -1 in effect encoding.  In the dummy encoding example, the city Bangalore at index 4  was encoded as 0000. Whereas in effect encoding it is represented by -1-1-1-1.
Let us see how we implement it in python-
```python
import category_encoders as ce
import pandas as pd
data=pd.DataFrame({'City':['Delhi','Mumbai','Hyderabad','Chennai','Bangalore','Delhi,'Hyderabad']}) encoder=ce.sum_coding.SumEncoder(cols='City',verbose=False,)

#Original Data
data
``
<img width="134" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/f8771b7f-1257-49aa-a8fb-988614cbf2bf">
```
encoder.fit_transform(data)
```
<img width="302" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/9715ddf2-af8d-4188-b6fe-4e4e454f708f">

### Feature Scaling
Feature scaling is the process of normalizing the range of features in a dataset.
Real-world datasets often contain features that are varying in degrees of magnitude, range, and units. Therefore, in order for machine learning models to interpret these features on the same scale, we need to perform feature scaling.
In the world of science, we all know the importance of comparing apples to apples, and yet many people, especially beginners, have a tendency to overlook feature scaling as part of their data preprocessing for machine learning. As we will see in this article, this can cause models to make predictions that are inaccurate.
Normalization
Normalisation, also known as min-max scaling, is a scaling technique whereby the values in a column are shifted so that they are bounded between a fixed range of 0 and 1.
MinMaxScaler is the Scikit-learn function for normalization.
Standardization
On the other hand, standardization or Z-score normalization is another scaling technique whereby the values in a column are rescaled so that they demonstrate the properties of a standard Gaussian distribution, that is mean = 0 and variance = 1.
StandardScaler is the Scikit-learn function for standardization.
Unlike StandardScaler, RobustScaler scales features using statistics that are robust to outliers. More specifically, RobustScaler removes the median and scales the data according to the interquartile range, thus making it less susceptible to outliers in the data.
Normalisation vs standardisation
Here comes the million-dollar question — when should we use normalization and when should we use standardization?
As much as I hate the response I’m about to give, it depends.
The choice between normalization and standardization really comes down to the application.
Standardization is generally preferred over normalization in most machine learning contexts as it is especially important when comparing the similarities between features based on certain distance measures. This is most prominent in Principal Component Analysis (PCA), a dimensionality reduction algorithm, where we are interested in the components that maximize the variance in the data.
Normalization, on the other hand, also offers many practical applications, particularly in computer vision and image processing where pixel intensities have to be normalized in order to fit within the RGB color range between 0 and 255. Moreover, neural network algorithms typically require data to be normalized to a 0 to 1 scale before model training.
At the end of the day, there is no definitive answer as to whether you should normalize or standardize your data. One can always apply both techniques and compare the model performance under each approach for the best result.
### Creating Interaction Features
The complex collaborative effects of features towards prediction of a variable is called feature interaction. Another aspect of feature interaction is the variation of one feature with respect to another with which it is interacting. These variables are often referred to as interaction variables.
Commonly, we encounter pairwise feature interactions in datasets where features interact in groups of 2. For example, the risk of developing a heart disease would depend on your BMI which is defined as weight / height². Here, { weight, height } is a pair-wise interaction. The less common higher-order feature interactions, where we see more than 2 features interacting, are common in the sciences where they have such complex relationships. For example, {x₁, x₂, x₃, x₄} is a 4th order interaction in log(x₁² + x₂ + x₃*x₄²).
Identifying the feature interactions present in your dataset can be useful for various reasons including:
Understanding the relationships between features in your dataset and its effect on the prediction and avoid biases from interpreting models with only the main effects and not the interactive effects
Using the information about interactions to explicitly build expressive models
Engineering features to improve model performance
## Feature Transformation
It is mandatory to digitize categorical features for models to work properly. This is called encoding.
The scale between features in the dataset can be very different from each other (or they may have different units). For example, while the “Age” feature varies between 0–100, “Car Price” can vary between 0–1000000. Some machine learning methods are affected by these scale differences, so normalizing the difference will contribute to the success of the model.
Models like KNN and SVM are distance-based algorithms means that the distance between points is used to obtain clusters or to find out similarities. The distance of unscaled features would also be unscaled and will be misleading the model. In addition, models using gradient descent optimization such as linear regression, logistic regression, or neural networks are also negatively affected by unscaled data. Coefficients of linear models are also affected by unscaled features. (In general, we don’t need to scale features in ensemble methods as the depth will probably not 
change.)
Transformation decreases the effects of outliers since the variability is reduced by scaling.
It improves model performance regarding the non-linear relationship between the target feature and the independent feature.
Some machine learning models are based on the assumption that the features are normally distributed. However, in real-life problems, the data is usually not normally distributed. In this case, we apply transformations to approximate these skewed data to the normal distribution so that the models can yield better results.

### Log Transformation
Log transformation is one of the most used Gaussian transformation methods. The log of each value is taken in feature, a nice way to deal with large numbers (Log of 1,000,000 is only 6). Thus, it reduces the impact of both high and low values in features.

It is used to approximate non-normally distributed data to the normal distribution. It is generally used for right-skewed features. Since it is logarithmic, it cannot be used for features that have negative values.
x_log = np.log(x)
The feature shown below is slightly right-skewed (upper charts). After the logarithmic transition, it is better at the point of Gaussian distribution.
<img width="518" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/567f9870-b1fc-4e31-a23f-5449abc5fc4d">


### Box-Cox Transformation
This is included in the concept of power transformations. Data must be positive. Lambda, λ is a value between -5 and 5. An optimal lambda value should be selected (like hyperparameter tuning).
T(Y) = (Y exp(λ)-1)/λ
We can use also scipy.stats to calculate lambda.
```
x_boxcox, lda = stat.boxcox(x)
print(lda)
#0.7964531473656952
```
<img width="555" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/5e828a18-f6ef-46e0-b15a-5d94b5a31c23">

### Principal Component Analysis (PCA)
PCA is a linear dimensionality reduction technique which converts a set of correlated features in the high dimensional space into a series of uncorrelated features in the low dimensional space. These uncorrelated features are also called principal components. PCA is an orthogonal linear transformation which means that all the principal components are perpendicular to each other. It transforms the data in such a way that the first component tries to explain maximum variance from the original data. It is an unsupervised algorithm i.e. it does not take into consideration the class labels.
<img width="419" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/19a92651-54bc-49c4-a00a-c8553bab698e">
The above figure shows two features F1 and F2 plotted in a two-dimensional space. As we can see from the plot above, the variance on feature F2 is much higher than the variance on feature F1 which means
information preserved by F2 >> information preserved by F1
Now suppose we want to convert this 2D data into 1D, we need to drop one feature. We know that feature F1 preserves much less information than F2, so we can safely drop feature F1 and use feature F2 as our final feature.
Now, let’s look at one more plot
<img width="438" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/cc726302-7ad1-4a12-b589-ce23d76b52b0">
Here, the variance on feature F2 is equal to the variance on feature F1 which means
Information preserved by F2 = information preserved by F1
now if we want to reduce the dimensions of the data from 2D to 1D we cannot drop one feature as we lose a lot of information.
So, what can we do here?
Here, PCA comes into the picture
<img width="425" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/f7b00b82-6d95-4b80-b483-407469dab7a9">
PCA performs linear orthogonal transformation on the data to find features F1’ and F2’ such that the variance on F1’ >> variance on F2’.

## Advanced Feature Engineering Techniques
1. Resampling Imbalanced Data
In practice, you will encounter imbalanced data more often than not. This does not necessarily have to be a problem if your target only has a slight imbalance. You could then resolve it by using proper validation measures for the data such as Balanced Accuracy, Precision-Recall Curves or F1-score.
Unfortunately, this is not always the case and your target variable might be highly imbalanced (e.g., 10:1). Instead, you can oversample the minority target in order to introduce balance using a technique called SMOTE.
SMOTE
SMOTE stands for Synthetic Minority Oversampling Technique and is an oversampling technique used to increase the samples in a minority class.
It generates new samples by looking at the feature space of the target and detecting nearest neighbors. Then, it simply selects similar samples and changes a column at a time randomly within the feature space of the neighboring samples.
The module to implement SMOTE can be found within the imbalanced-learn package. You can simply import the package and apply a fit_transform:.
```python
import pandas as pd
from imblearn.over_sampling import SMOTE

# Import data and create X, y
df = pd.read_csv('creditcard_small.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1].map({1:'Fraud', 0:'No Fraud'})

# Resample data
X_resampled, y_resampled = SMOTE(sampling_strategy={"Fraud":1000}).fit_resample(X, y)
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
```
<img width="625" alt="image" src="https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/2d2216c7-a094-4487-8937-fc4f25fb2916">
As you can see the model successfully oversampled the target variable. There are several strategies that you can take when oversampling using SMOTE:

'minority': resample only the minority class;
'not minority': resample all classes but the minority class;
'not majority': resample all classes but the majority class;
'all': resample all classes;
When dict, the keys correspond to the targeted classes. The values correspond to the desired number of samples for each targeted class.
I chose to use a dictionary to specify the extent to which I wanted to oversample my data.
Additional tip 1: If you have categorical variables in your dataset SMOTE is likely to create values for those variables that cannot happen. For example, if you have a variable called isMale, which could only take 0 or 1, then SMOTE might create 0.365 as a value.
Instead, you can use SMOTENC which takes into account the nature of categorical variables. This version is also available in the imbalanced-learnpackage.
Additional tip 2: Make sure to oversample after creating the train/test split so that you only oversample the train data. You typically do not want to test your model on synthetic data.
2. Creating New Features
To improve the quality and predictive power of our models, new features from existing variables are often created. We can create some interaction (e.g., multiply or divide) between each pair of variables hoping to find an interesting new feature. This, however, is a lengthy process and requires a significant amount of coding. Fortunately, this can be automated using Deep Feature Synthesis.
Deep Feature Synthesis
Deep feature synthesis (DFS) is an algorithm which enables you to quickly create new variables with varying depth. For example, you can multiply pairs of columns but you can also choose to first multiply Column A with Column B and then add Column C.
First, let me introduce the data I will be using for the example. I have chosen to use HR analytics data since the features are easily interpretable:
![1_aLsjksR5yZ0TlMMbMshkkw](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/e6f70bff-c799-4d7f-9684-6a66697cafc2)
Simply based on our intuition we could identify average_monthly_hours divided by number_project as an interesting new variable. However, there might be much more relationships that we miss out on if we only follow intuition.
The package does require an understanding of their use of Entities. However, if you use a single table you can simply follow the code below:
```python
import featuretools as ft
import pandas as pd

# Create Entity
turnover_df = pd.read_csv('turnover.csv')
es = ft.EntitySet(id = 'Turnover')
es.entity_from_dataframe(entity_id = 'hr', dataframe = turnover_df, index = 'index')

# Run deep feature synthesis with transformation primitives
feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'hr',
                                      trans_primitives = ['add_numeric', 'multiply_numeric'], 
                                      verbose=True)
```
The first step is to create an entity from which relationships can be created with other tables if necessary. Next, we can simply run ft.dfs in order to create new variables. We specify how variables are created with the parameter trans_primitives. We chose to either add numeric variables together or multiply.
![1_NmYQho56xc4JtA_ZBfncZA](https://github.com/arastuthakur/365_Days_Of_Machine_Learning/assets/76399951/34ebacde-9113-4034-ae5f-7ec9c86f55d9)
As you can see in the image above we have created an additional 668 features using only a few lines of code. A few examples of the features that were created:
last_evaluation multiplied with satisfaction_level
left multiplied with promotion_last_5years
average_monthly_hours multiplied with satisfaction_level plus time_spend_company
Additional tip 1: Note that the implementation here is relatively basic. The great thing about DFS is that it can create new variables from aggregations between tables (e.g., facts and dimensions). See this link for an example.
Additional tip 2: Run ft.list_primitives()in order to see the full list of aggregation that you can do. It even handles timestamps, null values, and long/lat information.

## Resources
Explore further resources to deepen your understanding of Feature Engineering.

## Books

1. **"Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists"** by Alice Zheng and Amanda Casari - A comprehensive book that covers various aspects of feature engineering with practical examples.

2. **"Practical Feature Engineering"** by Dipanjan Sarkar - This book focuses on hands-on feature engineering techniques and best practices for real-world machine learning projects.

## Online Courses

3. **Coursera - "Feature Engineering for Machine Learning"** - A course that dives deep into feature engineering techniques and strategies, taught by leading experts in the field.

4. **edX - "Feature Engineering for Improving Learning Environments"** - An online course that explores feature engineering for educational data and personalized learning.

## Additional Reading

5. **[Feature Engineering for Machine Learning: A Comprehensive Overview](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)** - An article that provides a comprehensive overview of feature engineering techniques and their importance in machine learning.

6. **[Feature Engineering: A Comprehensive Overview (Kaggle)](https://www.kaggle.com/getting-started/116173)** - Kaggle's guide on feature engineering, covering various techniques and examples.

7. **[The Caret Package for Machine Learning in R](https://topepo.github.io/caret/feature-engineering.html)** - A resource on feature engineering using the caret package in R, including practical examples.

8. **[Feature Engineering Tips, Tricks, and Recipes (GitHub)](https://github.com/Feature-engineering-data)** - A collection of feature engineering tips, tricks, and code snippets on GitHub.

## GitHub Repositories

9. **[Feature Engineering Recipes for Machine Learning](https://github.com/alicezheng/feature-engineering-book)** - A GitHub repository containing code examples and notebooks related to feature engineering techniques.

10. **[Feature Engineering for Machine Learning](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/feature%20engineering/Feature%20Engineering%20for%20Machine%20Learning%20-%20Preliminaries.ipynb)** - A notebook with practical feature engineering examples from the book "Practical Feature Engineering."

These resources cover a wide range of topics related to Feature Engineering, from books and courses to articles, code examples, and practical tips. Whether you're looking to enhance your feature engineering skills or start from scratch, these materials can be valuable in your journey.
