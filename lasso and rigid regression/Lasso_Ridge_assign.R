######################## PROBLEM 1 ##################################
# Load the Data set 
startup <- read.csv(file.choose())
# Reorder the variables
startup <- startup[,c(2,1,3,4,5)]

install.packages("glmnet")
library(glmnet)

x <- model.matrix(Profit ~ ., data = startup)[,-1]
y <- startup$Profit

grid <- 10^seq(10, -2, length = 100)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

############################# PROBLEM 2 ##############################

# Load the Data set 
comp_data <- read.csv(file.choose())
# Reorder the variables
comp_data <- comp_data[,c(2,1,3,4,5,6,7,8,9,10,11)]

install.packages("glmnet")
library(glmnet)

x <- model.matrix(price ~ ., data = comp_data)[,-1]
y <- comp_data$price

grid <- 10^seq(10, -2, length = 100)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

######################### PROBLEM 3 ####################################

# Load the Data set 
car_data <- read.csv(file.choose())

#feature selection
cols <- c('Price','Age_08_04', 'KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight')
car_data <- car_data[cols]

install.packages("glmnet")
library(glmnet)

x <- model.matrix(Price ~ ., data = car_data)[,-1]
y <- car_data$Price

grid <- 10^seq(10, -2, length = 100)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

######################## PROBLEM 4 #####################################

# Load the Data set 
life_data <- read.csv(file.choose())
# Reorder the variables
life_data <- life_data[,c(4,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)]

sum(is.na(life_data))
life_data <-  na.omit(life_data)


install.packages("glmnet")
library(glmnet)

x <- model.matrix(Life_expectancy ~ ., data = life_data)[,-1]
y <- life_data$Life_expectancy

grid <- 10^seq(10, -2, length = 1000)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)


cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

