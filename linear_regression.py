import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import mglearn



boston = load_boston()

print(boston.keys())

print(type(boston))
# <class 'sklearn.utils.Bunch'>

print(boston.data.shape)
# 506 rows with 13 features per observation

print(boston.feature_names)
# 'CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'

print(boston.DESCR)
# - CRIM     per capita crime rate by town
# - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
# - INDUS    proportion of non-retail business acres per town
# - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# - NOX      nitric oxides concentration (parts per 10 million)
# - RM       average number of rooms per dwelling
# - AGE      proportion of owner-occupied units built prior to 1940
# - DIS      weighted distances to five Boston employment centres
# - RAD      index of accessibility to radial highways
# - TAX      full-value property-tax rate per $10,000
# - PTRATIO  pupil-teacher ratio by town
# - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# - LSTAT    % lower status of the population
# - MEDV     Median value of owner-occupied homes in $1000's



###################################################################################################
# Create primary dataframe and assign titles
###################################################################################################

bos = pd.DataFrame(boston.data)    # create a new df called bos
bos.columns = boston.feature_names    # reassign colums names from the boston.features
bos['PRICE'] = boston.target       # Add price as a new column name
print(bos.head())

print(bos.describe())
# print statistical summary of boston housing dataset

# Divide datasets into output and input datasets
X = bos.drop('PRICE', axis = 1)   # Predicted value (y_hat)
Y = bos['PRICE']                  # All x values (x_i)

# Divide into four main test sub-arrays and print array size
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


######################################################################################################################
#  Simple Linear Regression
######################################################################################################################

# assign a linear model to the object "lm"
lm = LinearRegression()
lm.fit(X_train, Y_train)


Y_pred = lm.predict(X_test)
print(Y_pred)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()



coefs = pd.DataFrame(lm.coef_.transpose())
print(coefs.transpose())


print(boston.feature_names)
coefs.columns = boston.feature_names


# print coefficients, intercept, mse and R2 values
# print("model coefficients: ", coefs, "\n")
print("Model Intercept: {:.3f}".format(lm.intercept_))
print("Mean Square Error: {:.3f}".format(sklearn.metrics.mean_squared_error(Y_test, Y_pred)))
print("Training Set Score: {:.3f}".format(lm.score(X_train, Y_train)))
print("Training Set Score: {:.3f}".format(lm.score(X_test, Y_test)))