

##########################################################################################################
# EVALUATING MULTIPLE ML MODELS (TITANIC DATASET)
##########################################################################################################

# Notes:
# - This code demonstrates how to evaluate the performance of multi-classifier on Kaggle's infamous Titanic datasest.
# - Becuase Kaggle datasets are given as a pair of "train" (known y values) and "test" (unknown y values),
# we will sample out of test dataset, so that classifier accuracy can be asessed.
# - This demo highlights the benifits of using Scikit-learn's shared model functions ("__name__, fit, predict)



# TODO: Convert .py files to Jupyter notebook and replace on Github

# TODO: fix chained indexing
# TODO: Add plotly plots
# TODO: Add ROC curve overlay for each classifier
# TODO: Create two different bar charts for test and train data (verify that the sss is creating identical test samples)
# TODO: Add plotly barchart/gantt chart to show the timeit runtime for each model (optomized at its highest setting)




##########################################################################################################
# SETUP
##########################################################################################################


# Load libraries:
import numpy as np
import pandas as pd
import re as re
import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.offline
import plotly.graph_objs as go


#settings:
np.set_printoptions(threshold=np.nan)

# Load classifiers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp



##########################################################################################################
# LOAD DATSSET
##########################################################################################################

# open csv train file as a pandas array (splitting will take place after feature engineering)
dataset = pd.read_csv('datasets/train.csv', header = 0, dtype={'Age': np.float64})




##########################################################################################################
# FEATURE ENGINEERING
##########################################################################################################

# Verification Printouts:
# print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
# print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
# print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
# print(train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
# print(pd.crosstab(train['Title'], train['Sex']))  # create a cross table to view the frequency of each title within each gender category
# print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())  # table showing how title matches to each title


# Create features for "is alone", , fill NA's in fare and embark, and create categorical fare set
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 # siblings + parents + self
dataset['IsAlone'] = 0  # Create a dummy variable for if family member is alone
dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1  # If family size ==1, assign is alone to 1
dataset['Embarked'] = dataset['Embarked'].fillna('S') # Fill NA's in the embarked category with the most common origin point (S: Southampton)
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median()) # Fill in NA's in the "Fare" column with the median price (14.4542)
dataset['CategoricalFare'] = pd.qcut(dataset['Fare'], 4)  # Create new element in train dataset using pd.qcut()

# creating features in both the train and test datasets (which are contained in the list full_data
age_avg = dataset['Age'].mean() # set the mean age to age_avg (29.6991)
age_std = dataset['Age'].std()  # set the std of age to age_std (14.1812)
age_null_count = dataset['Age'].isnull().sum()  # set the number of null values in age to age_null_count
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count) # make a random list of ages ranging within 1 sd from the mean, size equals null count
dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list # Assign the random data list to the NA elements in the "Age" category:
dataset['Age'] = dataset['Age'].astype(int)  # cast "Age" category to interger
dataset['CategoricalAge'] = pd.cut(dataset['Age'], 5)  # Create new feature of categorical age


# create a new function to extract title from a string runs that match "xxx. "string
# group(1) allows us to access the actual string that we are targeting

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:	# If the title exists, extract and return it.
        return title_search.group(1)
    else:
        return ""  # else, return an empty string


dataset['Title'] = dataset['Name'].apply(get_title)
dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')     # replace the rare terms with the word "rare" in the title category
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')  # replace the misspelled titles   ( correct title can be confirmed via gender and table (e.g. Mlle and Mme both corespond to female gendered observations))
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')





# Mapping Sex
dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)  # rename the sex feature with dummy variables (0 for female and 1 for male)


# Mapping titles
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}  # Mapping the title with a
dataset['Title'] = dataset['Title'].map(title_mapping)
dataset['Title'] = dataset['Title'].fillna(0)


# Mapping Embarked
dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


# Mapping Fare
dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3


# Mapping Age
dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
dataset.loc[dataset['Age'] > 64, 'Age'] = 4




##########################################################################################################
# FEATURE SELECTION
##########################################################################################################



# Feature Selection  #TODO: Consider showing correlation between each column and survival rate (to back up intuition)

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize', 'CategoricalAge', 'CategoricalFare']
train = dataset.drop(drop_elements, axis=1)
train = train.values  # needs to be recast to allow slicing  (once done, train.values is inaccessible)

# print(type(train))
# print(train, "\n")






# note: Parameters roughly optimized by hand

classifiers = [
    KNeighborsClassifier(4),
    SVC(probability=True, C=1.0),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(n_estimators=100, max_depth=15),
    AdaBoostClassifier(n_estimators=100),
    GradientBoostingClassifier(learning_rate=0.30, max_depth=3),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 100), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False),

    ]



log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)   # create an empty df to populate
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0) #create the sss object later to be used to split ( note that it doesnt take the sample as a parameter)


# Convention Note: (use upper case letters ("X"/"Y") for matrices and lower case letters ("x"/"y") for vectors)
X = train[0:, 1:]   # Assign everything but the first column to a matrix called X
y = train[0:, 0]    # Assign the first column from the training dataset (i.e. survical boolean) to a vector called y


acc_dict = {}  # dictinary to hold the accuracy of each classifier


for train_index, test_index in sss.split(X, y):  # Test dataset must be derived from the train file since we don't know the actual values of the train file.
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__  # built-in name eg. (KNeighborsClassifier)
        clf.fit(X_train, y_train)  # fit each classifier object (feeding the train matrix (X inputs) and solutions (y vector)
        train_predictions = clf.predict(X_test)  # predict with each classifier using the X_test files
        acc = accuracy_score(y_test, train_predictions) #temp value acc to hold the accuracy

        if name in acc_dict:  # if built-in matches the dictionary name
            acc_dict[name] += acc  # add accuracy to names value (0 + acc)
        else:
            acc_dict[name] = acc  #else accuracy == 0 (right?)


print(acc_dict)

for clf in acc_dict:  # clf = classifier name (KNeighborsClassifier, GaussianNB)
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)  # the actual table that holds the classifier accuracies


print(log)  # printing the classifier's comparative performance

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
plt.show()
