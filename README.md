## Data Cleaning and preparation
During the cleaning of the data we found no missing, null or duplicated values. The data is clean and ready to be used for analysis.

We inspected the data and found that there was a file format issue with the 'date' column. The feature was dropped due to its lack of relevance to the analysis.

## Data Exploration
We found that the data had a lot of correlation between the features. The most important correlation was between the 'price' and 'sqft_living' features.
Due to the high correlation between the features, we decided to keep as many as possible, being that most of them had a p-value lower than 0.05

Furthermore we found the data consisted of many outliers, which we decided to keep. We based our decision on
the fact that the data was real estate data, and that the outliers most likely are real data. 
We did however remove the most extreme outliers, which we defined as houses over 600 square meters, because
they were very few and very extreme.

# Results:

### Results for Linear Regression
**Train R2:**  0.6935641365311427 \
**Test R2:** 0.6929698293624038 \
**RMSE Test:**  215443.29128718175 \
**RMSE Train:**  200088.5323205967

Shows indication of overfitting. 

### Results for MultiLinear Regression
**R2 score** - 1 would be the best:  0.6948638067062136 \
\
**RMSE train:**  188874.65229578203 \
About 188874 units away from the actual values in the traning data. \
**RMSE test:**  186544.82177273842 \
About 186544 units away from the actual values in the test data. \
\
Root Mean Squared Error: 188874.65229578203 \
Gives an idea in dollars of the difference in prediction an actual value:
Gives high weight to large errors - useful when large errors are particularly undesirable \
\
**Mean Absolute Error:** 121200.61938364392
The average difference between the predicted and actual values is 121200 \
The measure gives an equal weight to all errors, whether they are small or big.\
**Mean Absolute Percentage Error:** 24.429329272767784 \
The average percentage difference between the predicted and actual values is 24.42%

### Results for Polynominal Regression
**R2 score:**  0.867465450526247 \
**Train score:**  0.867465450526247 \
**Test score:**  0.8672596391153224 \
\
**RMSE train:**  124269.28792199031 \
About 124269 units away from the actual values in the training data.\
**RMSE test:** 124574.25909990243
About 124574 units away from the actual values in the test data. \
\
The RMSE-scores are quite close, which suggests that the model has a good balance between bias and variance.
It should generalize well to new data.\
\
**Mean Absolute Error:** 82717.98519660125\
The average difference between the predicted and actual values is 82717
**Mean Absolute Percentage Error:** 16.836124684535438
The average percentage difference between the predicted and actual values is 16.83%

