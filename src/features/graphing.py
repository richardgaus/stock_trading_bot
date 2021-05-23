import seaborn as sns
import pandas as pd
from matplotlib import pyplot

sns.set_theme(style="darkgrid")


# DataFrame[input_label][day] = correlation on that day
def prepare_graph_data(lag_corr, output_label, input_labels):
    data = pd.DataFrame(columns=['lag', 'corr', 'feature'], index=range(len(lag_corr) * len(lag_corr[0].columns)))
    i = 0
    for lag, df in enumerate(lag_corr):
        for input_feature in df.loc[output_label].keys():
            if input_feature not in input_labels: continue
            data.loc[i]['corr'] = df.loc[output_label][input_feature]
            data.loc[i]['feature'] = input_feature
            data.loc[i]['lag'] = lag
            i += 1
    return data.dropna().astype({'lag': 'int32', 'corr': 'float64', 'feature': 'object'}, )


def graph_lag_correlation(lag_corr, output_label, input_labels):
    data = prepare_graph_data(lag_corr, output_label, input_labels)

    sns.lineplot(x="lag", y="corr",
                 hue="feature",
                 data=data)
