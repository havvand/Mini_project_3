#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as stm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
import sklearn.metrics as sm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, scale
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('data/house-data.csv')
print(df.shape)
print(df.sample)

#%% Clean and prepare the data
df.info()
# No nan values
print(df.isna().sum())
# No missing values
print(df.notnull().sum())
# No duplicates
print(df.duplicated().sum())

#%%
import seaborn as sns
df['sqm_living'] = df['sqft_living'] * 0.092903
df['sqm_living15'] = df['sqft_living15'] * 0.092903
df = df[df['sqm_living'] <= 600].drop(['sqft_living'], axis=1)
df = df[df['sqm_living15'] <= 600].drop(['sqft_living15'], axis=1)

# Plot the distribution of the 'price' column
sns.displot(df['sqm_living'], kde=True)

# Show the plot
plt.show()
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
X = df.drop(['id', 'price', 'date', 'sqft_basement', 'sqft_above'], axis=1)
y = df[target]

# Add a constant to the DataFrame for the intercept
X = stm.add_constant(X)

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Standardize features
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

#%% Run a baseline regression model
cv = KFold(n_splits=10, random_state=42, shuffle=True)

# Linear regression
lin_reg = LinearRegression().fit(X_train_scaled, y_train)
lr_score_train = -1 * cross_val_score(lin_reg, X_train_scaled, y_train, cv=cv, scoring='neg_root_mean_squared_error').mean()
lr_core_test = mean_squared_error(y_test, lin_reg.predict(X_test_scaled), squared=False)

#%% Generate Principal Components
pca = PCA()
X_train_pc = pca.fit_transform(X_train_scaled)

pd.DataFrame(pca.components_.T).loc[:4, :]

#%% Dertermine the number of components
lin_reg = LinearRegression()
rmse_list = []

for i in range(1, X_train_pc.shape[1] + 1):
    rmse_score = -1 * cross_val_score( lin_reg,
                                       X_train_pc[:, :i],
                                       y_train,
                                       cv=cv,
                                       scoring='neg_root_mean_squared_error').mean()
    rmse_list.append(rmse_score)

plt.plot(rmse_list, marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("RMSE")
plt.xlim(xmin = -1)
plt.xticks(np.arange(X_train_pc.shape[1]), np.arange(1, X_train_pc.shape[1] + 1)) # Set x-ticks to min value of 1 and max value of number of principal components.
plt.axhline(y=lr_score_train, color='g', linestyle='-') # Draw a horizontal line at the value of the baseline RMSE
plt.show()


#%% Run PC Regression
best_number_of_components = 15

lin_reg_pca = LinearRegression().fit(X_train_pc[:, :best_number_of_components], y_train)

pcr_score_train = -1 * cross_val_score(lin_reg_pca,
                                       X_train_pc[:, :best_number_of_components],
                                       y_train,
                                       cv=cv,
                                       scoring='neg_root_mean_squared_error').mean()

lin_reg_pc = LinearRegression().fit(X_train_pc[:, :best_number_of_components], y_train)

X_test_pc = pca.transform(X_test_scaled)[:,:best_number_of_components]

train_preds = lin_reg_pc.predict(X_train_pc[:, :best_number_of_components])

preds = lin_reg_pc.predict(X_test_pc)

# Scatter plot of actual vs predicted values
plt.scatter(y_test, preds, color='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

pcr_score_test = mean_squared_error(y_test, preds, squared=False)

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, preds)

test_RMSE = mean_squared_error(y_test, preds, squared=False)
train_RMSE = mean_squared_error(y_train, train_preds, squared=False)

print('Train R2 ', train_r2)
print('Test R2', test_r2)

print('RMSE Test: ', test_RMSE)
print('RMSE Train: ', train_RMSE)



#%%
def olsi(data, f_cols):
    X = data[f_cols]
    y = df['price']
    X = stm.add_constant(X)
    #fit linear regression model
    model = stm.OLS(y, X).fit()
    return model

model = olsi(X, X.columns)
print(model.summary())