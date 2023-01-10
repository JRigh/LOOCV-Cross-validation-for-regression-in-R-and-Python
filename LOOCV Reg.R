#-------------------------------------------------------------
# Leave-one-out Cross-validation (LOOCV) for regression models
# in R
#-------------------------------------------------------------

# 1. generate artificial data
set.seed(2023)
n <- 300
x <- rnorm(n = n, mean = 5, sd = 3)
y <- x^2 + rnorm(n = n, mean = 0, sd = 8)
data = data.frame(x = x, y = y)

# 2. plot the models that we want to fit
par(mfrow = c(1,2), pty = "s")

model1 <- lm(y ~ x)
plot(x = x, y = y, main = 'Linear model', cex = 1.1, pch = 1, lwd = 1.2)
yhat1 <- model1$coef[1] + model1$coef[2] * x
lines(x, yhat1, lw = 2.5, col = 'red')

model2 <- lm(y ~ x + I(x^2))
plot(x = x, y = y, main = 'Quadratic model', cex = 1.1, pch = 1, lwd = 1.2)
yhat2 <- model2$coef[1] + model2$coef[2] * x  + model2$coef[3] * x^2
lines(x[order(x)], yhat2[order(x)], lw = 2.5, col = 'red')

# 3. Cross-Validation
# fit the models on leave-one-out samples
pred.cv.mod1 <- pred.cv.mod2 <- numeric(n)

for(i in 1:n) {
  
  # quadratic model
  mod1 = lm(y ~ x, subset = -i)
  pred.cv.mod1[i] = predict(mod1, data[i,])
  
  # quadratic model
  mod2 = lm(y ~ x + I(x^2), subset = -i)
  pred.cv.mod2[i] = predict(mod2, data[i,])
}

MSE1 = (1/n) * sum((y - pred.cv.mod1)^2) # theta_hat_pe
MSE2 = (1/n) * sum((y - pred.cv.mod2)^2) # theta_hat_pe

# Root Mean Squared Error (RMSE)
sqrt(c(MSE1, MSE2))
# [1] 15.68599  7.99332
# The second model (Quadratic) has the lowest RMSE and thus is prefered.

#----
# end
#----
