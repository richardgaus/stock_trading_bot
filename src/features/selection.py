from sklearn.preprocessing import RobustScaler
import numpy as np

# read in data
import pandas as pd
import graphing

def assess_p_correlation(corr_table_per_lag, output_labels, data_labels) -> [str]:
    print('selecting highest correlations...')
    input_features = set(data_labels) - set(output_labels)
    max_corr = pd.DataFrame(data=np.zeros((len(output_labels), len(input_features))), columns=input_features,
                            index=set(output_labels))
    max_corr_lags = {}

    for lag, corr_for_lag in enumerate(corr_table_per_lag):
        if (lag == 0): continue
        # corr_for_lag is a correlation matrix between all features for the lag
        output_cols = corr_for_lag[output_labels]
        for output_feature in output_cols:
            for input_feature, input_feature_corr in output_cols[output_feature].items():
                if (input_feature not in max_corr_lags.keys()): max_corr_lags[input_feature] = {}
                if input_feature_corr != 0 and input_feature in data_labels and input_feature not in output_labels and abs(
                        input_feature_corr) > max_corr[input_feature][output_feature]:
                    max_corr[input_feature][output_feature] = abs(input_feature_corr)
                    max_corr_lags[input_feature][output_feature] = lag

    # for each input feature, take maximum correlation to output
    max_corr_reduced = {}
    for input_feature in max_corr:
        max_corr_reduced[input_feature] = max_corr[input_feature].max()

    sorted_features = {k: v for k, v in sorted(max_corr_reduced.items(), key=lambda item: item[1])}
    return sorted_features


# define feature selection
# fs = SelectKBest(score_func=f_regression, k=10)
# apply feature selection
# X_selected = fs.fit_transform(rows, output_data)

def normalize(stock_data):
    print('normalizing...')
    transformer = RobustScaler(quantile_range=(5.0, 95.0)).fit(stock_data)
    return pd.DataFrame(transformer.transform(stock_data), columns=stock_data.columns)


def relativize(stock_data, ignore_columns=None):
    print('relativizing...')
    if ignore_columns is None:
        ignore_columns = []

    relative_data = stock_data.copy()
    for index, row in stock_data.iterrows():
        if index == 0 or (stock_data.loc[index - 1] == 0).all(): continue
        relative_data.loc[index] = (stock_data.loc[index] - stock_data.loc[index - 1]) / stock_data.loc[index - 1]

    relative_data[ignore_columns] = stock_data[ignore_columns]
    return relative_data


DAYS_LAG = 14

# Returns dict: { feature_name: max_correlation }
def select_relevant_features(stock_data, output_labels: [str]):
    stock_data = normalize(stock_data)
    # stock_data = relativize(stock_data, ['sentiment'])

    data_labels = stock_data.columns.tolist()
    input_data = stock_data.drop(output_labels, axis=1)
    output_data = stock_data[output_labels]
    output_data_relative = output_data.copy()

    rows = input_data.values
    labels = input_data.columns
    input_labels = input_data.columns
    input_labels_number = len(input_labels)

    # time_correlation[10] - p values for each column when shifted 10 days back
    corr_table_per_lag = []

    print('calculating correlations for different time lags...')
    for i in range(0, DAYS_LAG):
        # corr = f_regression(input_data, output_data)
        # corr = np.corrcoef(np.transpose(stock_data))
        corr = abs(stock_data.corr())
        corr_table_per_lag.append(corr)
        stock_data[input_labels] = stock_data[input_labels].shift(1)
        stock_data = stock_data.fillna(0)

    features = assess_p_correlation(corr_table_per_lag, output_labels, data_labels)
    return corr_table_per_lag, features

if __name__ == '__main__':
    data = pd.read_csv('../../data/raw/historical_price_data_wol.csv', sep=';')
    aapl_sentiment = pd.read_csv('../../data/raw/aapl_sentiment_news.csv').rename(columns={'Datetime': 'date'})

    # Join stock data and sentiment
    data['date'] = pd.to_datetime(data["date"])
    aapl_sentiment['date'] = pd.to_datetime(aapl_sentiment["date"])
    data = data.merge(aapl_sentiment, on='date')

    data = data.sort_values(by='date')
    data = data.drop(['label', 'acceptedDate', 'reportedCurrency', 'period', 'symbol', 'date', 'fillingDate'], axis=1)

    corr, features = (select_relevant_features(data, ['high']))
    print('done.')

    print(features)