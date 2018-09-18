#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 18:17:54 2018

@author: Tianjing Cai
ANLY590 HW0
"""
# 1.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('Hitters.csv').dropna()
df = df.rename(columns = {'Unnamed: 0': 'Player'})
df.info()

y = df.Salary
X = df.drop(['Player','Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64')
X.info()


# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

def plot_coefs(coefs, name):
    plt.figure()
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('log alpha')
    plt.ylabel('coefficients')
    plt.title('coefficient trajectories for ' + name + ' regression at each alpha value')
    plt.legend(X.columns.values)
    
''' 
LASSO regression
'''

lasso = Lasso(max_iter=10000, normalize=True)
coefs_lasso = []
alphas = 10**np.linspace(10,-2,100)*0.5
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X), y)
    coefs_lasso.append(lasso.coef_)
plot_coefs(coefs_lasso, 'LASSO')

lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
lassocv.fit(X_train, y_train)
LASSO_best_alpha = lassocv.alpha_
lasso.set_params(alpha=LASSO_best_alpha) # fit LASSO regression with best alpha value after perform 10 folds CV
lasso.fit(X_train, y_train)
best_LASSO_MSE = mean_squared_error(y_test, lasso.predict(X_test))
LASSO_best_coefs = pd.Series(lasso.coef_, index=X.columns)
print("Best coefficients for LASSO regression: \n" , LASSO_best_coefs)   

### For the result, we could only see the last two remaining predictors or the last 4 remaining predictors. 
### The last four remaining predictors are Hits, Walks, CRuns and CRBI. When alpha = 0.049, all predictors remain.
### 
'''
Ridge regression 
'''
   
alphas = 10**np.linspace(10,-2,100)*0.5 # generate 100 alphas values
alphas
ridge = Ridge(normalize=True)
coefs_ridge = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs_ridge.append(ridge.coef_)

plot_coefs(coefs_ridge, 'ridge') # plot coefs for each alpha value


ridgecv = RidgeCV(alphas=alphas, scoring='mean_squared_error', normalize=True, cv = 10)
ridgecv.fit(X_train, y_train)
ridge_best_alpha = ridgecv.alpha_

ridge4 = Ridge(alpha=ridge_best_alpha, normalize=True) # fit ridge regression with best alpha value after perform 10 folds CV
ridge4.fit(X_train, y_train)
best_ridge_MSE = mean_squared_error(y_test, ridge4.predict(X_test))


ridge_best_coefs = pd.Series(ridge4.coef_, index=X.columns)
print("Best coefficients for ridge regression: \n" , ridge_best_coefs)

### The best penalty parameter for ridge regression is 0.0087.

# 2.
'''
The bias-variance tradeoff is the property of a set of predictive models whereby models with a lower bias in 
parameter estimation have a higher variance of the parameter estimates across samples, and vice versa. The role of regularization term
is used to prevent model from overfitting data, reducing variance. As we could see from both figures of LASSO and ridge regression, 
the smaller value of log alpha the larger value of each coefficients is and as log alpha value increase, the value for each coefficients decrease.
The larger value of each coefficient is, the more wigglier the model is (higer variance); on the other hand, the smaller value of each coefficient is, 
the less wigglier the model is (lower variance). Thus, the model variance decrease as value of log alpha increase; but the model bias increase at same time.
'''