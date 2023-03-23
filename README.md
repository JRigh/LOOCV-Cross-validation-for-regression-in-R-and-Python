## LOOCV-Cross-validation-for-regression
Leave-one-out Cross-validation for regression models

In the first example, we compare the fit among a Linear and a Quadratic model for artificially generated data. As shown in the R and Python output, the Quadratic model is 
a much better fit for the data as it minimizes the Root Mean Squared Error (RMSE), a medric for prediction error after cross-validation. 

The R output.

![plot1P](/assets/plot1R.png)

The Python output.

![plot1P](/assets/plot1P.png)

They are equivalent. Cross-validation in Python can be done as follows:

```python
import pandas as pd
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
```

Then using R, we compare the fit among three models for binary data on a real dataset, 'mtcars'. We try to predict the binary variable 'vs' which 
correspond to the engine type using 'mpg' or miles per gallon. We find that the Probit model minimizes the RMSE compared to a Logistic model and a Complementary
Log-log model.

![Rplot2](/assets/Rplot2.png)

The Python script for the second example is for now unfinished.
