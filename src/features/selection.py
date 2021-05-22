# pearson's correlation feature selection for numeric input and numeric output
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
import datetime
import numpy as np

# read in data
import pandas as pd

# clean data
data = pd.read_csv('../data/historical_price_data_wol.csv', sep=';')
data['date'] = pd.to_datetime(data["date"])
data = data.sort_values(by='date')
data = data.drop(['label', 'acceptedDate', 'reportedCurrency', 'period', 'symbol', 'date', 'fillingDate'], axis=1)
transformer = RobustScaler(quantile_range=(5.0, 95.0)).fit(data)
data = pd.DataFrame(transformer.transform(data), columns=data.columns)
data_labels = data.columns.tolist()
def label_to_idx(label):
    return data_labels.index(label)

# format input & output data
output_labels = ['high']
input_data = data.drop(output_labels, axis=1)
output_data = data[output_labels]
output_data_relative = output_data.copy()
for index, row in output_data.iterrows():
    if index == 0 or output_data['high'][index-1] == 0: continue
    output_data_relative['high'][index] = (output_data['high'][index] - output_data['high'][index-1]) / output_data['high'][index-1]

rows = input_data.values
labels = input_data.columns
input_labels = input_data.columns
input_labels_number = len(input_labels)


def assess_time_correlation(tc) -> [str]:
    max_pvals = [np.Infinity] * input_labels_number
    for i, day_correlation in enumerate(tc):
        feature_sum = 0
        for i, pval in enumerate(day_correlation[1]):
            if pval != 0 and pval < max_pvals[i]:
                max_pvals[i] = pval

    avg = sum(max_pvals) / len(max_pvals)
    sorted_indicies = np.argsort(max_pvals)
    print("above average features: \n")
    for i, idx in enumerate(sorted_indicies):
        if i > 3: break
        print("{}: {}".format(input_labels[idx], max_pvals[idx]))
    return max_pvals

def assess_p_correlation(tc) -> [str]:
    max_corr = [0] * (len(data_labels) + 1)
    # max_day
    for day, day_correlation in enumerate(tc):
        output_idx = label_to_idx(output_labels[0])
        output_row = day_correlation[output_idx]
        for feature_idx, corr in enumerate(output_row):
            if corr != 0 and feature_idx != output_idx and corr > max_corr[feature_idx]:
                max_corr[feature_idx] = corr

    sorted_indicies = np.argsort(max_corr, )
#     print("above average features: \n")
#     for i, idx in enumerate(reversed(sorted_indicies)):
#         # if i > 3: break
#         if idx > len(data_labels) or idx > len(max_corr): break
#         print("{}: {}".format(data_labels[idx], max_corr[idx]))
    return sorted_indicies




# time_correlation[10] - p values for each column when shifted 10 days back
time_correlation = []
for i in range(0, 14):
    # corr = f_regression(input_data, output_data)
    corr = np.corrcoef(np.transpose(data))
    time_correlation.append(corr)
    output_data_relative = pd.concat([pd.Series(0), output_data_relative])
    input_data = pd.concat([pd.DataFrame(0, index=np.arange(1), columns=labels), input_data])

# define feature selection
# fs = SelectKBest(score_func=f_regression, k=10)
# apply feature selection
# X_selected = fs.fit_transform(rows, output_data)

label_idx_sorted_by_corr = assess_p_correlation(time_correlation)
