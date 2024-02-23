from math import sqrt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error
import sklearn.metrics as sm
from sklearn import metrics
import statsmodels.api as smm

df = pd.read_csv('data/house-data.csv')
print(df.shape)
print(df.sample)

#%% Clean data
df.info()
# No missing values
print(df.notnull().sum())
# No duplicates
print(df.duplicated().sum())

#%% Describe the data
df_described = df.describe()
print(df_described)

#%% Coorelation matrix
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df['date'] = pd.to_datetime(df['date'])
corr_matrix = df.corr()
print(corr_matrix)
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

#%% Dropping houses over 600 sqm
df['sqm_living'] = df['sqft_living'] * 0.092903
df = df[df['sqm_living'] <= 600].drop(['sqft_living'], axis=1)
print(df.shape)
sns.pairplot(df, x_vars=['sqm_living', 'bathrooms', 'grade', 'condition'], y_vars='price', height=7, aspect=0.7, kind='reg')

#%% Drop columns
df_dropped = df.drop(['id', 'date', 'sqft_lot', 'sqft_lot15', 'sqft_basement'], axis=1)
print('Dropped shape: ', df_dropped.shape)

corr_matrix = df_dropped.corr()
plt.figure(figsize=(30, 30))
ax = sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", annot_kws={'size': 10})
 #Manually set the annotations
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j+0.5, i+0.5, round(corr_matrix.iloc[i, j], 2),
                       ha="center", va="center", color="black")
plt.show()

#%%
# Define the intervals
'''bins = np.arange(0, df_dropped['sqft_living'].max() + 500, 500)

# Convert the 'sqft_living' column to a categorical type with ordered categories
df_dropped['sqft_living'] = pd.cut(df_dropped['sqft_living'], bins=bins, include_lowest=True, right=False)

fig, ax = plt.subplots(figsize=(30, 30))
sns.boxplot(x='sqft_living', y='price', data=df_dropped, ax=ax)
ax.set_yticks(np.arange(min(df_dropped['price']), max(df_dropped['price']), 100000))
ax.set_xticks(np.arange(min(df_dropped['sqft_living']), max(df_dropped['sqft_living'])+1, 500))

plt.show()'''


#%% Multiple linear regression
feature_cols = df_dropped.columns.drop('price')
X = df_dropped[feature_cols]
y = df_dropped['price']

X.info()
y.info()

#%% Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

#%%
print('Intercept: ', lin_reg.intercept_)
print('Coefficients: ', lin_reg.coef_)

list(zip(feature_cols, lin_reg.coef_))


#%%
y_pred = lin_reg.predict(X_test)
y_test_pred = lin_reg.predict(X_train)
print(y_test_pred.shape)
print(y_train.shape)
#%%
# Calculating the RMSE on the training set
rmse_train = sqrt(mean_squared_error(y_test, y_pred))
print('\nRMSE train: ', rmse_train)
print('About 188874 units away from the actual values in the traning data.')
# Calculating the RMSE on the test set
rmse_test = sqrt(mean_squared_error(y_train, y_test_pred))
print('\nRMSE test: ', rmse_test)
print('About 186544 units away from the actual values in the test data. \n'
      'The test RMSE is higher then the train RMSE, could be a sign of overfitting.')

#%%
print('R2 score - 1 would be the best: ', r2_score(y_test, y_pred))
print(metrics.mean_absolute_error(y_test, y_pred))

# calculate MSE using scikit-learn
print('Mean Squared Error - an abstract number, that can be very high when predicting house prices:')
print(metrics.mean_squared_error(y_test, y_pred))

# calculate RMSE using scikit-learn
print('Root Mean Squared Error - squareroot of MSE. Gives an idea in dollars of the difference in prediction an actual value:')
print('Gives hight weight to large errors - useful when large errors are particularly undesirable')
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#%%
# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
print('The average difference between the predicted and actual values is 127558')
print('The measure gives an equal weight to all errors, whether they are small or big.')

# Calculate MAPE (Mean Absolute Percentage Error) which is the average percentage difference between the predicted and actual values
def mean_absolute_percentage_error(y_true, y_predi):
    y_true, y_predi = np.array(y_true), np.array(y_predi)
    return np.mean(np.abs((y_true - y_predi) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print('Mean Absolute Percentage Error:', mape)
print('The average percentage difference between the predicted and actual values is 24.96%')

#%%
# Visualise the regression results
plt.title('Multiple Linear Regression')
plt.scatter(y_test, y_pred, color='blue')
plt.show()
#%%
def olsi(data, f_cols):
    X = data[f_cols]
    y = data['price']
    X = smm.add_constant(X)
    model = smm.OLS(y, X).fit()
    return model

model = olsi(df_dropped, feature_cols)

print('AIC', model.aic)
model.summary()

