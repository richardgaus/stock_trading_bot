from keras.models import Model
from keras.layers import (
    Input, Masking, LSTM, Dropout, Permute, Conv1D, BatchNormalization,
    Activation, GlobalAveragePooling1D, concatenate, Dense,
    multiply, Reshape
)

def generate_model(num_features:int,
                   num_timesteps:int=365,
                   num_lstm_units:int=8,
                   dropout_probability:float=0.8):
    ip = Input(shape=(num_features, num_timesteps))

    x = Masking()(ip)
    x = LSTM(units=num_lstm_units)(x)
    x = Dropout(dropout_probability)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(units=6)(x)

    model = Model(ip, out)
    model.summary()

    return model

def squeeze_excite_block(input):
    """Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    """
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se