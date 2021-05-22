from sklearn.preprocessing import RobustScaler
import pandas as pd

data = pd.read_csv('../data/historical_price_data_wol.csv', sep=';')
transformer = RobustScaler().fit(data)
print(transformer.transform(data))