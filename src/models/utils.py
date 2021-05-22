from typing import List

from pandas import DataFrame
from keras.models import Model

from src.data.utils import train_validation_test_split

def train_model(model:Model,
                assets:List[DataFrame],
                num_timesteps:int=365,
                num_variables:int=5,
                learning_rate:float=1e-3):
    train_ts, valid_ts, test_ts = train_validation_test_split(
        assets=assets, ratio=(0.7, 0.15, 0.15)
    )

    optimizer = Adam(lr=learning_rate)