
# pearson's correlation feature selection for numeric input and numeric output
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.compose import ColumnTransformer
import datetime
import numpy as np

# read in data
import pandas as pd
data = pd.read_csv('data/energy/energydata_complete.csv')
data['date'] = data['date'].apply(lambda d : datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S').timestamp())

input_data = data.drop(['Appliances'], axis=1)
output_data = data['Appliances']
rows = input_data.values
labels = input_data.columns

# time_correlation[10] - p values for each column when shifted 10 days back
time_correlation = []
for i in range(0, 14):
    time_correlation.append(f_regression(input_data, output_data))
    output_data = pd.concat([pd.Series(0), output_data])
    input_data = input_data.append(pd.Series(name="nullrow-" + str(i)))
# generate dataset
X, y = make_regression(n_samples=100, n_features=100, n_informative=10)
# define feature selection
fs = SelectKBest(score_func=f_regression, k=10)
# apply feature selection
X_selected = fs.fit_transform(rows, output_data)



print(X_selected.shape)