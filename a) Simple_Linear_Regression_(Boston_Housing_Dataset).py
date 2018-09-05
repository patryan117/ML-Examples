



##########################################################################################################
# SIMPLE LINEAR REGRESSION ON BOSTON HOUSING DATASET
##########################################################################################################



##########################################################################################################
# SETUP
##########################################################################################################



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline

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



#######################################################################################################################
# Create dataframe and test / train datasets
#######################################################################################################################

bos = pd.DataFrame(boston.data)    # create a new df called bos
bos.columns = boston.feature_names    # reassign colums names from the boston.features
bos['PRICE'] = boston.target       # Add price as a new column name
print(bos.head())

print(bos.describe())
# print statistical summary of boston housing dataset

# Divide datasets into output and input datasets
X = bos.drop('PRICE', axis = 1)   # Output: Predicted value (y_hat)
Y = bos['PRICE']                  # Input: All x values (x_i)

# Divide into four main test sub-arrays and print array size
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


######################################################################################################################
#  Simple Linear Regression
######################################################################################################################


# assign a linear regression model to the object "lm"
lm = LinearRegression()
lm.fit(X_train, Y_train)

# Make predictions for the test_x variables and assign to Y_pred
Y_pred = lm.predict(X_test)


# Create a scatterplot of the predicted prices vs actual prices
trace = go.Scatter(
    x = Y_test,
    y = Y_pred,
    mode = 'markers'
)

layout= go.Layout(
    title= 'Predicted Versus Actual',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Actual',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Predicted',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)

data = [trace]

# Plot and embed in ipython notebook!
fig= go.Figure(data=data, layout=layout)
plotly.offline.plot(fig)


#
# l= []
# y= []
# data= pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")
# # Setting colors for plot.
# N= 53
# c= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]
#
# for i in range(int(N)):
#     y.append((2000+i))
#     trace0= go.Scatter(
#         x= data['Rank'],
#         y= data['Population']+(i*1000000),
#         mode= 'markers',
#         marker= dict(size= 14,
#                     line= dict(width=1),
#                     color= c[i],
#                     opacity= 0.3
#                    ),name= y[i],
#         text= data['State']) # The hover text goes here...
#     l.append(trace0);
#
# layout= go.Layout(
#     title= 'Stats of USA States',
#     hovermode= 'closest',
#     xaxis= dict(
#         title= 'Population',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
#     yaxis=dict(
#         title= 'Rank',
#         ticklen= 5,
#         gridwidth= 2,
#     ),
#     showlegend= False
# )
# fig= go.Figure(data=l, layout=layout)
# plotly.offline.plot(fig)


coef_array = (pd.DataFrame({'Name':lm.coef_,'Age':boston.feature_names}))



# print coefficients, intercept, mse and R2 values

print("#############################################################################################################")
print("Simple Linear Regression Test Results")
print("#############################################################################################################\n\n")

print("Model Intercepts: ", "\n", coef_array)
print("Mean Square Error: {:.3f}".format(sklearn.metrics.mean_squared_error(Y_test, Y_pred)))
print("Training Set Score: {:.3f}".format(lm.score(X_train, Y_train)))
print("Training Set Score: {:.3f}".format(lm.score(X_test, Y_test)))



# Notes:

# Linear model has not input parameters, and only minimizes the mean square error
# Linear models do particularly well on datasets with large features.
# We do not appear to be overfitting, so a regularized variant like Ridge or Lasso linear regression wont help.
