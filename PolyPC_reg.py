import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, scale
import statsmodels.api as sm

df = pd.read_csv('data/house-data.csv')
print(df.shape)
print(df.sample)

#%% Clean and prepare the data
df.info()
# No missing values
print(df.notnull().sum())
# No duplicates
print(df.duplicated().sum())

#%% Inspect data
df_described = df.describe()
print(df_described)

# Coorelation matrix
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df['date'] = pd.to_datetime(df['date'])
corr_matrix = df.corr()
print(corr_matrix)
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

#%% Drop columns and train the model
target = 'price'

X = df.drop(['id', 'price', 'date', 'sqft_lot', 'sqft_lot15', 'sqft_living', 'sqft_basement'], axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Fitting to Polynomial Regression
poly_model = PolynomialFeatures(degree=3)
X_poly = poly_model.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

#%%
# Predicting the training set results using the trained model using the polynomial features from the training set
y_pred = pol_reg.predict(X_poly)
# Transformning the test set X_test to polynomial features using poly_model
X_test_poly = poly_model.transform(X_test)
y_test_pred = pol_reg.predict(X_test_poly)

# Calculating the R2 score on the training set. How well does it match the actual values.
Train_Score = pol_reg.score(X_poly, y)
# Calculating the R2 score on the test set. How well does it match the actual values.
Test_Score = pol_reg.score(X_test_poly, y_test)

# Same as Train_Score but using the sklearn.metrics library
r2 = r2_score(y, y_pred)

print('R2 score: ', r2)
print('Train score: ', Train_Score)
print('Test score: ', Test_Score)

# Calculating the RMSE on the training set
rmse_train = sqrt(mean_squared_error(y, y_pred))
print('\nRMSE train: ', rmse_train)
print('About 129173 units away from the actual values in the traning data.')
# Calculating the RMSE on the test set
rmse_test = sqrt(mean_squared_error(y_test, y_test_pred))
print('\nRMSE test: ', rmse_test)
print('About 137985 units away from the actual values in the test data. \n'
      'The test RMSE is higher then the train RMSE, could be a sign of overfitting.')

#%% Evaluate the model


def olsi_poly(data, f_cols, degree):
    X = data[f_cols]
    y = data['price']

    # Generate polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    X_poly = sm.add_constant(X_poly)

    # Fit linear regression model
    model = sm.OLS(y, X_poly).fit()
    return model

model = olsi_poly(df, ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'], 2)
print(model.summary())