from typing import List

import os
import shutil
import numpy as np
from pandas import DataFrame
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from src.data.utils import train_validation_test_split

"""def train_model(model:Model,
                assets:List[DataFrame],
                num_timesteps:int=365,
                num_variables:int=5,
                learning_rate:float=1e-3):
    train_ts, valid_ts, test_ts = train_validation_test_split(
        assets=assets, ratio=(0.7, 0.15, 0.15)
    )

    optimizer = Adam(lr=learning_rate)"""

def train_model(model:Model,
                X_train:np.array,
                y_train:np.array,
                X_valid:np.array,
                y_valid:np.array,
                logdir_parent:str,
                learning_rate:float=1e-3,
                batch_size:int=128,
                num_epochs:int=500):
    # Tensorboard logging
    logdir = logdir_parent / f'lr{learning_rate}eps{num_epochs}bs{batch_size}'
    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
    tensorboard_callback = TensorBoard(log_dir=logdir)
    # Reduce learning rate on plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        patience=100,
        mode='auto',
        factor=1./np.cbrt(2),
        cooldown=0,
        min_lr=1e-4,
        verbose=2
    )
    callback_list = [tensorboard_callback, reduce_lr]
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mse'
    )
    model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=callback_list,
        verbose=2,
        validation_data=(X_valid, y_valid)
    )
