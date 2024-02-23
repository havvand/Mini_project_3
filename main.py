
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
import sklearn.metrics as sm
from sklearn import metrics
import statsmodels.api as sm

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

#%% Drop columns
df_dropped = df.drop(['id', 'date', 'sqft_lot', 'sqft_above' , 'condition', 'sqft_lot15', 'yr_renovated', 'zipcode', 'waterfront', 'sqft_living15'], axis=1)
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

#%%
sns.regplot(x='sqft_living', y='price', data=df_dropped, scatter_kws={'s': 10})
plt.show()

#%%
sns.regplot(x='lat', y='price', data=df_dropped, scatter_kws={'s': 10})
plt.show()

#%% Train the model
X = df_dropped['sqft_living'].values.reshape(-1, 1)
y = df_dropped['price'].values.reshape(-1, 1)

plt.ylabel('Price')
plt.xlabel('sqft_living')
plt.scatter(X, y, color='blue')
plt.show()

#%% Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.15)

# Print the shape of the training and test data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

my_reg = LinearRegression()
my_reg.fit(X_train, y_train)

a = my_reg.coef_
b = my_reg.intercept_
print('a: ', a)

y_predict = my_reg.predict(X_test)
print(y_predict)

plt.title("Linear Regression")
plt.scatter(X, y, color='blue')
plt.plot(X_train, a*X_train + b, color='red')
plt.plot(X_test, y_predict, color='green')
plt.xlabel('sqft_living')
plt.ylabel('Price')
plt.show()

R2 = my_reg.score(X, y)
print('R2: ', R2) # Value is 0.492798
print('R2 Score for test', r2_score(y_test, y_predict))

#%% Evaluate the model
# Mean absolute error shows the difference between the predicted and actual values, a score of 0 means no error.
print('Mean absolute error: ', sm.mean_absolute_error(y_test, y_predict))

#%% Use ployfit-method
# Train model - Build model from training data with ployfit
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.15)
model_ploy = np.polyfit(np.ravel(X_train), np.ravel(y_train), 1)
print(model_ploy) # prints the coefficients, 277.5 and -34963.5 which is the intercept and slope (slope*x + intercept)

a, b = model_ploy[0], model_ploy[1] # Here we assign the coefficients to a and b
test = np.polyfit(np.ravel(X_test), np.ravel(y_test), 1) # Here we test the model with the test data.
print(test) # The coefficients are different from the training because the test data is different

a1, b1 = test[0], test[1] # Here we assign the coefficients to a1 and b1, because we want to use them in the plot

y_predict = my_reg.predict(X_test)

plt.title('Linear Regression')
plt.scatter(X, y, color='green')
plt.plot(X_test, a1*X_test + b1, color='orange')
plt.plot(X_train, a*X_train + b, color='blue')

plt.xlabel('sqft_living')
plt.ylabel('Price')
plt.show()

R2 = my_reg.score(X, y)
print(R2)
print('R2 Score for test', r2_score(y_test, y_predict))

#%% Multiple linear regression
df_dropped['house_age'] = df_dropped['yr_built'].apply(lambda x: 2023 - x)

features = ['sqft_living', 'bathrooms', 'grade', 'house_age']
X = df_dropped[features]
y = df_dropped['price']

# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.15)

# Now X_train is a DataFrame and you can select columns from it
X_train = X_train[features]

my_reg = LinearRegression()
my_reg.fit(X_train, y_train)

# To get the coefficients of each feature the model for the user to see
list(zip(features, my_reg.coef_))

y_predict = my_reg.predict(X_test[features])
print(y_predict)

R2 = my_reg.score(X, y)
print('R2: ', R2) # Value is higher than the previous model = 0.605921

print(metrics.mean_absolute_error(y_test, y_predict)) ## MAE value is 147811.2631

# calculate MSE using scikit-learn
print(metrics.mean_squared_error(y_test, y_predict)) # MSE value is 56308704607.24809

# calculate RMSE using scikit-learn.
print(np.sqrt(metrics.mean_squared_error(y_test, y_predict))) # RMSE value is 237294.5524

# The explained variance score: 1 is perfect prediction.
eV = round(explained_variance_score(y_test, y_predict), 6)
print('Explained variance score ',eV )

# R-squared
R2 = my_reg.score(X, y)
print('R2: ', R2) # Value is 0.605921 for the model
print('R2 Score for test data', r2_score(y_test, y_predict))

plt.title('Multiple Linear Regression')
plt.scatter(y_test, y_predict, color='blue') # Test are true values, y_predict are predicted values. If the model was perfect, all points would be on the line y=x
plt.show()

#%%
def olsi(data, f_cols):
    X = data[f_cols]
    y = data['price']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

model = olsi(df_dropped, features)

print('AIC', model.aic)
model.summary()

