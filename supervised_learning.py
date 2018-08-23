import mglearn
import matplotlib.pyplot as plt
import scipy
import numpy as np

import matplotlib.pyplot as plt;




#2
# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
plt.show()
print("**************************************************")

#3
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
print("**************************************************")


#4
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): {}".format(cancer.keys()))
print("**************************************************")


#5
print("Shape of cancer data: {}".format(cancer.data.shape))
print("**************************************************")


#6
print("Sample counts per class:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("**************************************************")


#7
print("Feature names:\n{}".format(cancer.feature_names))
print("**************************************************")


#8
from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))
print("**************************************************")


#9
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
print("**************************************************")


#10
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()
print("**************************************************")


#11
mglearn.plots.plot_knn_classification(n_neighbors=14)
plt.show()
print("**************************************************")


#12
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("**************************************************")


#13
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
print("**************************************************")

#14
clf.fit(X_train, y_train)
