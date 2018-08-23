import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
# import statsmodels.api as sm  (throws error apparently)

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams

from sklearn.datasets import load_boston
boston = load_boston()


print(boston.keys())

print(boston)

#506 rows with 13 features per observation
print(boston.data.shape)


