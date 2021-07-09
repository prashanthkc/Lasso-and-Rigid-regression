# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:11:55 2021

@author: prashanth
"""

######################### PROBLEM 1 ######################################


# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
startup = pd.read_csv("F:/assignment/lasso and rigid regression/Datasets_LassoRidge/50_Startups (1).csv")

# Rearrange the order of the variables
startup = startup.iloc[:, [4, 0, 1 ,2, 3]]
startup.columns


#labelEncoding for categprical data
from sklearn import preprocessing
L_enc = preprocessing.LabelEncoder()
startup['State'] = L_enc.fit_transform(startup['State'])

# Exploratory data analysis:
startup.describe()
cols = {'R&D Spend':'RD_Spend', 'Marketing Spend':'Marketing_Spend'}
startup.rename(cols,axis=1,inplace = True)


# Correlation matrix 
a = startup.corr()
a

# EDA
a1 = startup.describe()
a1
# Sctter plot and histogram between variables
sns.pairplot(startup) # sp-hp, wt-vol multicolinearity issue

# Preparing the model on train data 
model_train = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + State', data = startup).fit()
model_train.summary()

# Prediction
pred = model_train.predict(startup)
# Error
resid  = pred - startup.Profit
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(startup.iloc[:, 1:], startup.Profit)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(startup.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(startup.iloc[:, 1:])

# Adjusted r-square
lasso.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((pred_lasso - startup.Profit)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(startup.iloc[:, 1:], startup.Profit)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(startup.columns[1:]))

rm.alpha

pred_rm = rm.predict(startup.iloc[:, 1:])

# Adjusted r-square
rm.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((pred_rm - startup.Profit)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(startup.iloc[:, 1:], startup.Profit) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(startup.columns[1:]))

enet.alpha

pred_enet = enet.predict(startup.iloc[:, 1:])

# Adjusted r-square
enet.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((pred_enet - startup.Profit)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(startup.iloc[:, 1:], startup.Profit)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(startup.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((lasso_pred - startup.Profit)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(startup.iloc[:, 1:], startup.Profit)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(startup.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((ridge_pred - startup.Profit)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(startup.iloc[:, 1:], startup.Profit)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(startup.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(startup.iloc[:, 1:], startup.Profit)

# RMSE
np.sqrt(np.mean((enet_pred - startup.Profit)**2))

################################## PROBLEM 2 ################################

# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
comp_data= pd.read_csv("F:/assignment/lasso and rigid regression/Datasets_LassoRidge/Computer_Data (1).csv")

# Exploratory data analysis:
comp_data.describe()
comp_data = comp_data.iloc[:,1:]

#labelEncoding for categprical data
from sklearn import preprocessing
L_enc = preprocessing.LabelEncoder()
comp_data['cd'] = L_enc.fit_transform(comp_data['cd'])
comp_data['multi'] = L_enc.fit_transform(comp_data['multi'])
comp_data['premium'] = L_enc.fit_transform(comp_data['premium'])


# Correlation matrix 
a = comp_data.corr()
a

# EDA
a1 = comp_data.describe()
a1
# Sctter plot and histogram between variables
sns.pairplot(comp_data) # sp-hp, wt-vol multicolinearity issue

# Preparing the model on train data 
model_train = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + trend', data = comp_data).fit()
model_train.summary()

# Prediction
pred = model_train.predict(comp_data)
# Error
resid  = pred - comp_data.price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(comp_data.iloc[:, 1:], comp_data.price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(comp_data.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(comp_data.iloc[:, 1:])

# Adjusted r-square
lasso.score(comp_data.iloc[:, 1:], comp_data.price)

# RMSE
np.sqrt(np.mean((pred_lasso - comp_data.price)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(comp_data.iloc[:, 1:], comp_data.price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(comp_data.columns[1:]))

rm.alpha

pred_rm = rm.predict(comp_data.iloc[:, 1:])

# Adjusted r-square
rm.score(comp_data.iloc[:, 1:], comp_data.price)

# RMSE
np.sqrt(np.mean((pred_rm - comp_data.price)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(comp_data.iloc[:, 1:], comp_data.price) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(comp_data.columns[1:]))

enet.alpha

pred_enet = enet.predict(comp_data.iloc[:, 1:])

# Adjusted r-square
enet.score(comp_data.iloc[:, 1:], comp_data.price)

# RMSE
np.sqrt(np.mean((pred_enet - comp_data.price)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(comp_data.iloc[:, 1:], comp_data.price)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(comp_data.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(comp_data.iloc[:, 1:], comp_data.price)

# RMSE
np.sqrt(np.mean((lasso_pred - comp_data.price)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(comp_data.iloc[:, 1:], comp_data.price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(comp_data.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(comp_data.iloc[:, 1:], comp_data.price)

# RMSE
np.sqrt(np.mean((ridge_pred - comp_data.price)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(comp_data.iloc[:, 1:], comp_data.price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(comp_data.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(comp_data.iloc[:, 1:], comp_data.price)

# RMSE
np.sqrt(np.mean((enet_pred - comp_data.price)**2))

#################################### PROBLEM 3 #################################


# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
car_data = pd.read_csv("F://assignment//lasso and rigid regression//Datasets_LassoRidge//ToyotaCorolla (1).csv", engine='python')


#using only required features
cols = ['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']
car_data= car_data[cols]
car_data.describe()


# Correlation matrix 
a = car_data.corr()
a

# EDA
a1 = car_data.describe()
a1
# Sctter plot and histogram between variables
sns.pairplot(car_data) # sp-hp, wt-vol multicolinearity issue

# Preparing the model on train data 
model_train = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = car_data).fit()
model_train.summary()

# Prediction
pred = model_train.predict(car_data)
# Error
resid  = pred - car_data.Price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(car_data.iloc[:, 1:], car_data.Price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(car_data.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(car_data.iloc[:, 1:])

# Adjusted r-square
lasso.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((pred_lasso - car_data.Price)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(car_data.iloc[:, 1:], car_data.Price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(car_data.columns[1:]))

rm.alpha

pred_rm = rm.predict(car_data.iloc[:, 1:])

# Adjusted r-square
rm.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((pred_rm - car_data.Price)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(car_data.iloc[:, 1:], car_data.Price) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(car_data.columns[1:]))

enet.alpha

pred_enet = enet.predict(car_data.iloc[:, 1:])

# Adjusted r-square
enet.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((pred_enet - car_data.Price)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(car_data.iloc[:, 1:], car_data.Price)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(car_data.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((lasso_pred - car_data.Price)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(car_data.iloc[:, 1:], car_data.Price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(car_data.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((ridge_pred - car_data.Price)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(car_data.iloc[:, 1:], car_data.Price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(car_data.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((enet_pred - car_data.Price)**2))

#################################### PROBLEM 4 ##############################

# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
life_data = pd.read_csv("F:/assignment/lasso and rigid regression/Datasets_LassoRidge/Life_expectencey_LR.csv")

#removing na values
life_data.isnull().sum()
life_data = life_data.dropna()

# Rearrange the order of the variables
life_data = life_data.iloc[:, [3, 0, 1 ,2, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
life_data.columns


#labelEncoding for categprical data
from sklearn import preprocessing
L_enc = preprocessing.LabelEncoder()
life_data['Country'] = L_enc.fit_transform(life_data['Country'])
life_data['Status'] = L_enc.fit_transform(life_data['Status'])



# Correlation matrix 
a = life_data.corr()
a

# EDA
a1 = life_data.describe()
a1
# Sctter plot and histogram between variables
sns.pairplot(life_data) # sp-hp, wt-vol multicolinearity issue

# Preparing the model on train data 
model_train = smf.ols('Life_expectancy~ Country + Year + Status + Adult_Mortality + infant_deaths +Alcohol+percentage_expenditure + Hepatitis_B+Measles+BMI+under_five_deaths+Polio+Total_expenditure+Diphtheria+HIV_AIDS+GDP+Population+thinness+thinness_yr+Income_composition+Schooling', data = life_data).fit()
model_train.summary()

# Prediction
pred = model_train.predict(life_data)
# Error
resid  = pred - life_data.Life_expectancy
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(life_data.iloc[:, 1:], life_data.Life_expectancy)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(life_data.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(life_data.iloc[:, 1:])

# Adjusted r-square
lasso.score(life_data.iloc[:, 1:], life_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_lasso - life_data.Life_expectancy)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(life_data.iloc[:, 1:], life_data.Life_expectancy)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(life_data.columns[1:]))

rm.alpha

pred_rm = rm.predict(life_data.iloc[:, 1:])

# Adjusted r-square
rm.score(life_data.iloc[:, 1:], life_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_rm - life_data.Life_expectancy)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(life_data.iloc[:, 1:], life_data.Life_expectancy) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(life_data.columns[1:]))

enet.alpha

pred_enet = enet.predict(life_data.iloc[:, 1:])

# Adjusted r-square
enet.score(life_data.iloc[:, 1:], life_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_enet - life_data.Life_expectancy)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(life_data.iloc[:, 1:], life_data.Life_expectancy)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(life_data.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(life_data.iloc[:, 1:], life_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((lasso_pred - life_data.Life_expectancy)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(life_data.iloc[:, 1:], life_data.Life_expectancy)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(life_data.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(life_data.iloc[:, 1:], life_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((ridge_pred - life_data.Life_expectancy)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(life_data.iloc[:, 1:], life_data.Life_expectancy)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(life_data.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(life_data.iloc[:, 1:], life_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((enet_pred - life_data.Life_expectancy)**2))
