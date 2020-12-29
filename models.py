from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2D,PReLU, ReLU, Softmax, add,\
    Flatten, MaxPooling2D, Dense, Reshape, Input, Dropout, concatenate, GaussianNoise
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.utils
from keras.regularizers import l2,l1
from tensorflow.keras import initializers

Lidar2D = Sequential([
    Input(shape=(20, 200, 1)),
    GaussianNoise(0.005),
    Conv2D(5, 3, 1, padding='same', kernel_initializer=initializers.HeUniform),
    BatchNormalization(axis=3),
    PReLU(shared_axes=[1, 2]),
    Conv2D(5, 3, 1, padding='same', kernel_initializer=initializers.HeUniform),
    BatchNormalization(axis=3),
    PReLU(shared_axes=[1, 2]),
    Conv2D(5, 3, 2, padding='same', kernel_initializer=initializers.HeUniform),
    BatchNormalization(axis=3),
    PReLU(shared_axes=[1, 2]),
    Conv2D(5, 3, 1, padding='same', kernel_initializer=initializers.HeUniform),
    BatchNormalization(axis=3),
    PReLU(shared_axes=[1, 2]),
    Conv2D(5, 3, 2, padding='same', kernel_initializer=initializers.HeUniform),
    BatchNormalization(axis=3),
    PReLU(shared_axes=[1, 2]),
    Conv2D(1, 3, (1, 2), padding='same', kernel_initializer=initializers.HeUniform),
    BatchNormalization(axis=3),
    PReLU(shared_axes=[1, 2]),
    Flatten(),
    Dense(16, activation='relu'),
    # Dropout(0.7),
    Dense(256, activation='softmax'),
])


GPS = Sequential([
    Input(shape=(3)),
    Dense(50, activation='relu',kernel_regularizer=l2(l2=1e-4),bias_regularizer=l2(1e-4)),
    Dense(50, activation='relu',kernel_regularizer=l2(l2=1e-4),bias_regularizer=l2(1e-4)),
    Dense(256, activation='softmax'),
])

def multiModalNet():
    input_lid = Input(shape=(20, 200, 1))
    layer=GaussianNoise(0.005)(input_lid)
    layer = Conv2D(16, kernel_size=(5, 5), activation='linear', padding="SAME")(layer)
    layer=BatchNormalization(axis=3)(layer)
    layer=ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer)
    layer = Conv2D(8, kernel_size=(5, 5), activation='linear', padding="SAME")(layer)
    layer = BatchNormalization(axis=3)(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer)
    layer = Conv2D(4, kernel_size=(3, 3), activation='linear', padding="SAME")(layer)
    layer = BatchNormalization(axis=3)(layer)
    layer = ReLU()(layer)
    layer = Conv2D(1, kernel_size=(3, 3), activation='linear', padding="SAME")(layer)
    layer = BatchNormalization(axis=3)(layer)
    layer = ReLU()(layer)
    out_lid = Flatten()(layer)

    input_coord = Input(shape=(3))
    layer=GaussianNoise(0.002)(input_coord)
    out_coord = Dense(10, activation='relu')(layer)

    concatenated = concatenate([out_lid, out_coord])
    layer = Dense(50, activation='relu', kernel_regularizer=l2(l2=1e-4), bias_regularizer=l2(1e-4))(concatenated)
    layer = Dense(50, activation='relu',kernel_regularizer=l2(l2=1e-4),bias_regularizer=l2(1e-4))(layer)
    layer = Dense(50, activation='relu', kernel_regularizer=l2(l2=1e-4), bias_regularizer=l2(1e-4))(layer)
    predictions= Dense(256, activation='softmax',kernel_regularizer=l2(l2=1e-4),bias_regularizer=l2(1e-4))(layer)
    architecture = Model(inputs=[input_lid,input_coord], outputs=predictions)
    return architecture

def multiModalNetOLD():
    input_lid=Input(shape=(20, 200, 1))
    input_coord=Input(shape=(3))

    layer = Conv2D(32, kernel_size=(5, 5), activation='relu', padding="SAME")(input_lid)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer)
    layer = Conv2D(32, kernel_size=(5, 5), activation='relu', padding="SAME")(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer)
    layer = Conv2D(32, kernel_size=(5, 5), activation='relu', padding="SAME")(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer)
    layer = Conv2D(16, kernel_size=(3, 3), activation='relu', padding="SAME")(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer)
    layer = Conv2D(16, kernel_size=(3, 3), activation='relu', padding="SAME")(layer)
    layer = Flatten()(layer)
    out_lid = Dense(400, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer)

    layer = Dense(128, activation='relu')(input_coord)
    out_coord =GaussianNoise(0.002)(layer)

    concatenated = concatenate([out_lid, out_coord])
    reg_val = 0.001
    layer = Dense(600, activation='relu', kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(concatenated)
    layer = Dense(600, activation='relu', kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(layer)
    layer = Dense(500, activation='relu', kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(layer)
    predictions= Dense(256, activation='softmax')(layer)
    architecture = Model(inputs=[input_lid,input_coord], outputs=predictions)
    return architecture

