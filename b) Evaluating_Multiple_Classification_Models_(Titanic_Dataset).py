

##########################################################################################################
# EVALUATING MULTIPLE ML MODELS (TITANIC DATASET)
##########################################################################################################

# test
# asdfjkadsfjklfadkjldfs
# sadfjkldfsakjlfdsjkl
# fdjkladfskjldfsakjlfds


##########################################################################################################
# SETUP
##########################################################################################################


# Load libraries:
import numpy as np
import pandas as pd
import re as re  # regular exporessions
import matplotlib.pyplot as plt
import seaborn as sns

# Load classifiers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# load test and train datasets (from kaggles.s built in features)
train = pd.read_csv('datasets/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('datasets/test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test]


#ways to inspect the dataframes in pandas
# print(type(train))
# print(train.info())
# print(train.head())
# print("\n")

# # taking the Pclass and Survived categories from sample, and showing the mean
# print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
# print("\n")
# # Taking the sex and survived categories from sample and presenting the mean
# print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
# print("\n")


##########################################################################################################
# FEATURE ENGINEERING
##########################################################################################################




# Create family size feature
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 # siblings + parents + self
# print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
# print("\n")

# Create a dummy variable for if family member is alone
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
# Single passengers were more likley to have survived
print("\n")

# Fill NA's in the embarked category with the most common origin point (S: Southampton)
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
print("\n")



# Fill in NA's in the "Fare" column with the median price (14.4542)
# Create new element in train dataset using pd.qcut()
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# print(train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


#TODO: fix chained indexing

# creating features in both the train and test datasets (which are contained in the list full_data
for dataset in full_data:


    # set the mean age to age_avg (29.6991)
    age_avg = dataset['Age'].mean()

    # set the std of age to age_std (14.1812)
    age_std = dataset['Age'].std()

    # set the number of null values in age to age_null_count
    age_null_count = dataset['Age'].isnull().sum()

    # make a random list of ages ranging within 1 sd from the mean, size equals null count
    #TODO what the duck?  (why do this instead of applying the mean like we did with age?)

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    # Assign the random data list to the NA elements in the "Age" category:
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    # cast "Age" category to interger
    dataset['Age'] = dataset['Age'].astype(int)

# Create new feature of categorical age (only only in the train dataset)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
# print(train['CategoricalAge'])


# TODO figure out how this works
# print a table showing how categorical age influenced the survival outcome
# print(train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
print("\n")





# create a new function to extract title from a string runs that match "xxx. "string
# group(1) allows us to access the actual string that we are targeting
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:	# If the title exists, extract and return it.
        return title_search.group(1)
    else:
        return ""  # else, return an empty string

#TODO does apply do it one at a time
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

# create a cross table to view the frequency of each title within each gender category
# print(pd.crosstab(train['Title'], train['Sex']))



for dataset in full_data:

    # replace the rare terms with the word "rare" in the title category
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    # replace the mispelled titles
    # correct title can be inferred by gender and table (i.e. Mlle and Mme both corespond to female gendered datapoints)
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


#TODO: seperate rare into rare-male and rare-female
# table showing how title matches to each/ title
# print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# rename the sex feature with dummy variables (0 for female and 1 for male)
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4



# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', \
                 'Parch', 'FamilySize']

train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

test = test.drop(drop_elements, axis=1)

# print(train.head(10))

train = train.values
test = test.values



#TODO Classifier parameters can be retested here, but they should be auto-adjusted using a loop!!!)
#TODO Add plotly printout to show the timeit runtime for each model (optomized at its highest setting!!!)


# note: parameters optimized by hand

classifiers = [
    KNeighborsClassifier(4),
    SVC(probability=True, C=1.0),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(n_estimators=100, max_depth=15),
    AdaBoostClassifier(),
    GradientBoostingClassifier(learning_rate=0.30, max_depth=3),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

#TODO Include deep learning, hidden Markov and

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)   # create an empty df to populate

# TODO, why are we sampling anything?  shouldn't we just be applying the test data set?

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)  #create the sss object later to be used to split

# TODO summarize stratified sampling

X = train[0::, 1::]  #TODO: explain double slicing syntax?
y = train[0::, 0]

# print("X file:", X)  #TODO what is this?

#TODO add trial and test accuracy

acc_dict = {}

#TODO revisit (why are there splits going on if we already have the test file?)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__  # built-in name eg. (KNeighborsClassifier)
        clf.fit(X_train, y_train)  # fit each classifier object
        train_predictions = clf.predict(X_test)  # predict with each classifier using the X_test files
        acc = accuracy_score(y_test, train_predictions) #temp value acc to hold the accuracy


#TODO: Theres gotta be a better way to do this!

        if name in acc_dict:  # if built-in matches the dictionary name
            acc_dict[name] += acc  # add accuracy to names value (0 + acc)
        else:
            acc_dict[name] = acc  #else accuracy == 0 (right?)

# print(acc_dict)  (why are the accuracy entries on a scale of 1 - 10?)

for clf in acc_dict:  # clf = classifier name (KNeighborsClassifier, GaussianNB)

    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)  # the actual table that holds the classifier accuracies


print(log)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
plt.show()
