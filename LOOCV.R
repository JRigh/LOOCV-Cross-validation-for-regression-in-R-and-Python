#-------------------------------------------------------------
# Leave-one-out Cross-validation (LOOCV) for regression models
# in R
#-------------------------------------------------------------

#--------------------------------------------------
# Example 1: Linear and Quadratic regression models
#--------------------------------------------------

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

#---------------------------------------
# Example 2: binomial regressions models
#---------------------------------------

# splitting the dataset into training and test sets
data("mtcars")
head(mtcars)
set.seed(2023)
ind <- sample(2, nrow(mtcars), replace=TRUE, prob=c(0.6,0.4))
training <- mtcars[ind==1,]
testing <- mtcars[ind==2,]

# save a copy of entire dataset, training and testing datasets in .csv
write.csv(mtcars, 
          "C:/Users/julia/OneDrive/Desktop/github/16. Crossvalidation/mtcars.csv",
          row.names = FALSE)
write.csv(training, 
          "C:/Users/julia/OneDrive/Desktop/github/16. Crossvalidation/mtcars_training.csv",
          row.names = FALSE)
write.csv(testing, 
          "C:/Users/julia/OneDrive/Desktop/github/16. Crossvalidation/mtcars_testing.csv",
          row.names = FALSE)

# 2. Cross-Validation
# fit the models on leave-one-out samples
pred.cv.modl <- pred.cv.modp <- pred.cv.modc <- numeric(length=nrow(testing))

for(i in 1:nrow(testing)) {
  
  # logistic model
  modl = glm(vs ~ mpg, data=mtcars,
             family= binomial(link = "logit"), subset = -i)
  pred.cv.modl[i] = predict(modl, testing[i,])
  
  # probit model
  modp = glm(vs ~ mpg, data=mtcars,
             family= binomial(link = "probit"), subset = -i)
  pred.cv.modp[i] = predict(modp, testing[i,])
  
  # complementary log-log model
  modc = glm(vs ~ mpg, data=mtcars,
             family= binomial(link = "cloglog"), subset = -i)
  pred.cv.modc[i] = predict(modc, testing[i,])
}

MSE1 = (1/nrow(testing)) * sum((testing$vs - pred.cv.modl)^2) # theta_hat_pe
MSE2 = (1/nrow(testing)) * sum((testing$vs - pred.cv.modp)^2) # theta_hat_pe
MSE3 = (1/nrow(testing)) * sum((testing$vs - pred.cv.modc)^2) # theta_hat_pe

# Root Mean Squared Error (RMSE)
sqrt(c(MSE1, MSE2, MSE3))
# [1] 2.720540 1.485185 1.705154

# The second model (Quadratic) has the lowest RMSE and thus is prefered.

# 3. Plot of the fit on testing dataset with annotation
# logistic fit
modl <- glm(vs ~ mpg, data=training,
            family= binomial(link = "logit"))

# probit fit
modp <- glm(vs ~ mpg, data=training,
            family= binomial(link = "probit"))

# complementary log-log fit
modc <- glm(vs ~ mpg, data=training,
            family= binomial(link = "cloglog"))

# complete the dataset with predictions for each model
testing$pred.modl <- predict(modl, type = 'response', newdata = testing)
testing$pred.modp <- predict(modp, type = 'response', newdata = testing)
testing$pred.modc <- predict(modc, type = 'response', newdata = testing)

# plot 
ggplot(testing, aes(x = mpg, y = vs)) + 
  geom_point(size = 1.8) +
  geom_line(size = 1, data = testing, aes(x = mpg[order(testing$mpg)], 
                                          y = pred.modl[order(testing$pred.modl)],
                                          color='Logistic')) +
  geom_line(size = 1, data = testing, aes(x = mpg[order(testing$mpg)], 
                                          y = pred.modp[order(testing$pred.modp)],
                                          color='Probit')) +
  geom_line(size = 1, data = testing, aes(x = mpg[order(testing$mpg)], 
                                          y = pred.modc[order(testing$pred.modc)],
                                          color='Log-log')) +
  annotate('text', label = paste('Logistic RMSE = ', round(sqrt(MSE1),2)), x = 25, y = 0.65, size = 3) + 
  annotate('text', label = paste('Probit RMSE = ', round(sqrt(MSE2),2)), x = 25, y = 0.58, size = 3) + 
  annotate('text', label = paste('C Log-log RMSE = ', round(sqrt(MSE3),2)), x = 25, y = 0.51, size = 3) + 
  labs(title = 'Scatterplot - Fit of Logistic, Probit and Log-log models',
       subtitle = 'mtcars dataset', color = "Legend",
       y="Engine (0 = V-shaped, 1 = straight)", x="Miles/(US) gallon") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

#----
# end
#----