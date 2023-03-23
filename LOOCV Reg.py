#-------------------------------------------------------------
# Leave-one-out Cross-validation (LOOCV) for regression models
# in Python
#-------------------------------------------------------------

#--------------------------------------------------
# Example 1: Linear and Quadratic regression models
#--------------------------------------------------

# 1. generate artificial data
import numpy as np
np.random.seed(2023)

n = 300
x = np.random.normal(loc = 5, scale = 3, size = n)
y = x**2 + np.random.normal(loc = 0, scale = 8, size = n)

# 2. plot the models that we want to fit
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression

px = 1/plt.rcParams['figure.dpi'] 
plt.figure(figsize=(850*px, 400*px))
plt.subplot(1, 2, 1)
plt.plot(x, y, 'o', fillstyle = 'none', color = 'black')
plt.title('Linear model', fontsize = 15)
plt.xlabel("x") ; plt.ylabel("y")
beta1, beta0 = np.polyfit(x, y, 1)
yhat1 = beta0 + beta1*x
plt.plot(x, yhat1, color = 'red')

plt.subplot(1, 2, 2)
plt.plot(x, y, 'o', fillstyle = 'none', color = 'black')
plt.xlabel('x') ; plt.ylabel('y') ; plt.title('Quadatic model', fontsize = 15)
beta2, beta1, beta0 = np.polyfit(x, y, 2)
yhat2 = beta0 + beta1*x + beta2*(x**2)
orders = np.argsort(x.ravel())
plt.plot(x[orders], yhat2[orders], color = 'red')


# 3. Cross-Validation
# fit the models on leave-one-out samples
import pandas as pd
data = pd.DataFrame({'x': x, 'y': y})
xn = data['x'].values.reshape(-1,1)
yn = data['y'].values.reshape(-1,1)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import LeaveOneOut, cross_val_score
loocv1 = LeaveOneOut()

# linear model
mod1 = PolynomialFeatures(degree = 1, include_bias = False).fit_transform(xn)
mod11 = LinearRegression().fit(mod1, yn)

loocv1 = LeaveOneOut()
scoresmod1 = cross_val_score(mod11, 
                         mod1,
                         yn, 
                         scoring = 'neg_mean_squared_error',
                         cv = loocv1)

    
# quadratic model
mod2 = PolynomialFeatures(degree = 2, include_bias = False).fit_transform(xn)
mod22 = LinearRegression().fit(mod2, yn)

loocv2 = LeaveOneOut()
scoresmod2 = cross_val_score(mod22, 
                         mod2,
                         yn, 
                         scoring = 'neg_mean_squared_error',
                         cv = loocv2)

# Root Mean Squared Error (RMSE)
import statistics
import math

RMSE1 = math.sqrt(statistics.mean(abs(scoresmod1)))
RMSE2 = math.sqrt(statistics.mean(abs(scoresmod2)))
[RMSE1, RMSE2]
# [16.169293289892607, 7.873829105930071]
# The second model (Quadratic) has the lowest RMSE and thus is prefered.

#---------------------------------------
# Example 2: binomial regressions models
#---------------------------------------

# importing the data from .csv file

training = pd.read_csv("C:/Users/julia/OneDrive/Desktop/github/16. Crossvalidation/mtcars_training.csv")
testing = pd.read_csv("C:/Users/julia/OneDrive/Desktop/github/16. Crossvalidation/mtcars_testing.csv")
testing

# 2. Cross-Validation
# fit the models on leave-one-out samples
from sklearn.linear_model import LogisticRegression

# logistic model
modl = LogisticRegression(random_state=0).fit_transform(training['mpg'])
modl1 = LogisticRegression.fit(mod1, training['vs'])
  
#-----------
# unfinished
#-----------
