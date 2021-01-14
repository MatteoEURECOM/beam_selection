from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2D,PReLU, ReLU, Softmax, add,\
    Flatten, MaxPooling2D, Dense, Reshape, Input, Dropout, concatenate, GaussianNoise
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.utils
from keras.regularizers import l2,l1
from tensorflow.keras import initializers

def LIDAR(FLATTENED,LIDAR_TYPE):
    '''
    LIDAR Neural Network
    '''
    if(LIDAR_TYPE=='CENTERED'):
        if(FLATTENED):
            input_lid = Input(shape=(67, 67, 1))
        else:
            input_lid = Input(shape=(67, 67, 10))
    elif(LIDAR_TYPE=='ABSOLUTE'):
        if(FLATTENED):
            input_lid = Input(shape=(20, 200, 1))
        else:
            input_lid = Input(shape=(20, 200, 10))
    elif(LIDAR_TYPE=='ABSOLUTE_LARGE'):
        if(FLATTENED):
            input_lid = Input(shape=(60, 330, 1))
        else:
            input_lid = Input(shape=(60, 330, 10))
    layer=Conv2D(5, 3, 1, padding='same', kernel_initializer=initializers.HeUniform)(input_lid)
    layer=BatchNormalization(axis=3)(layer)
    layer=PReLU(shared_axes=[1, 2])(layer)
    layer=Conv2D(5, 3, 1, padding='same', kernel_initializer=initializers.HeUniform)(layer)
    layer=BatchNormalization(axis=3)(layer)
    layer=PReLU(shared_axes=[1, 2])(layer)
    layer=Conv2D(5, 3, 2, padding='same', kernel_initializer=initializers.HeUniform)(layer)
    layer=BatchNormalization(axis=3)(layer)
    layer=PReLU(shared_axes=[1, 2])(layer)
    layer=Conv2D(5, 3, 1, padding='same', kernel_initializer=initializers.HeUniform)(layer)
    layer=BatchNormalization(axis=3)(layer)
    layer=PReLU(shared_axes=[1, 2])(layer)
    layer=Conv2D(5, 3, 2, padding='same', kernel_initializer=initializers.HeUniform)(layer)
    layer=BatchNormalization(axis=3)(layer)
    layer=PReLU(shared_axes=[1, 2])(layer)
    layer=Conv2D(1, 3, (1, 2), padding='same', kernel_initializer=initializers.HeUniform)(layer)
    layer=BatchNormalization(axis=3)(layer)
    layer=PReLU(shared_axes=[1, 2])(layer)
    layer=Flatten()(layer)
    layer=Dense(16, activation='relu')(layer)
    predictions=Dense(256, activation='softmax')(layer)
    architecture = Model(inputs=input_lid, outputs=predictions)
    return architecture

def GPS():
    '''
    GPS Neural Network
    '''
    reg_val=0
    input_lid = Input(shape=(3))
    layer= Dense(85, activation='relu',kernel_regularizer=l2(reg_val),bias_regularizer=l2(reg_val))(input_lid)
    layer = Dense(85, activation='relu',kernel_regularizer=l2(reg_val),bias_regularizer=l2(reg_val))(layer)
    predictions =Dense(256, activation='softmax')(layer)
    architecture = Model(inputs=input_lid, outputs=predictions)
    return architecture

def MULTIMODAL(FLATTENED,LIDAR_TYPE):
    '''
    Multimodal Neural Network GPS+LIDAR
    '''
    if(LIDAR_TYPE=='CENTERED'):
        if(FLATTENED):
            input_lid = Input(shape=(67, 67, 1))
        else:
            input_lid = Input(shape=(67, 67, 10))
    elif(LIDAR_TYPE=='ABSOLUTE'):
        if(FLATTENED):
            input_lid = Input(shape=(20, 200, 1))
        else:
            input_lid = Input(shape=(20, 200, 10))
    elif(LIDAR_TYPE=='ABSOLUTE_LARGE'):
        if(FLATTENED):
            input_lid = Input(shape=(60, 330, 1))
        else:
            input_lid = Input(shape=(60, 330, 10))
    '''LIDAR branch'''
    #layer=GaussianNoise(0.01)(input_lid)
    layer = Conv2D(8, kernel_size=(5, 5), activation='linear', padding="SAME")(input_lid)
    layer=BatchNormalization(axis=3)(layer)
    layer=ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer)
    layer = Conv2D(4, kernel_size=(3, 3), activation='linear', padding="SAME")(layer)
    layer = BatchNormalization(axis=3)(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(layer)
    layer = Conv2D(2, kernel_size=(3, 3), activation='linear', padding="SAME")(layer)
    layer = BatchNormalization(axis=3)(layer)
    layer = ReLU()(layer)
    layer = Conv2D(1, kernel_size=(3, 3), activation='linear', padding="SAME")(layer)
    layer = BatchNormalization(axis=3)(layer)
    layer = ReLU()(layer)
    out_lid = Flatten()(layer)
    '''GPS branch'''
    input_coord = Input(shape=(3))
    #layer=GaussianNoise(0.01)(input_coord)
    out_coord = Dense(10, activation='relu')(input_coord)
    '''Concatenation'''
    concatenated = concatenate([out_lid, out_coord])
    reg=0.001
    layer = Dense(50, activation='relu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(concatenated)
    layer = Dense(50, activation='relu',kernel_regularizer=l2(reg),bias_regularizer=l2(reg))(layer)
    predictions= Dense(256, activation='softmax',kernel_regularizer=l2(reg),bias_regularizer=l2(reg))(layer)
    architecture = Model(inputs=[input_lid,input_coord], outputs=predictions)
    return architecture

def MULTIMODAL_OLD(FLATTENED,LIDAR_TYPE):
    if(LIDAR_TYPE=='CENTERED'):
        if(FLATTENED):
            input_lid = Input(shape=(67, 67, 1))
        else:
            input_lid = Input(shape=(67, 67, 10))
    elif(LIDAR_TYPE=='ABSOLUTE'):
        if(FLATTENED):
            input_lid = Input(shape=(20, 200, 1))
        else:
            input_lid = Input(shape=(20, 200, 10))
    elif(LIDAR_TYPE=='ABSOLUTE_LARGE'):
        if(FLATTENED):
            input_lid = Input(shape=(60, 330, 1))
        else:
            input_lid = Input(shape=(60, 330, 10))
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
    '''GPS branch'''
    input_coord = Input(shape=(3))
    layer = Dense(128, activation='relu')(input_coord)
    out_coord =GaussianNoise(0.002)(layer)
    '''Concatenation'''
    concatenated = concatenate([out_lid, out_coord])
    reg_val = 0
    layer = Dense(600, activation='relu', kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(concatenated)
    layer = Dense(600, activation='relu', kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(layer)
    layer = Dense(500, activation='relu', kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(layer)
    predictions= Dense(256, activation='softmax')(layer)
    architecture = Model(inputs=[input_lid,input_coord], outputs=predictions)
    return architecture

Lidar2D = Sequential([
    Input(shape=(20, 200, 1)),
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


def MIXTURE(FLATTENED, LIDAR_TYPE):
    if(LIDAR_TYPE=='CENTERED'):
        if(FLATTENED):
            input_lid = Input(shape=(67, 67, 1))
        else:
            input_lid = Input(shape=(67, 67, 10))
    elif(LIDAR_TYPE=='ABSOLUTE'):
        if(FLATTENED):
            input_lid = Input(shape=(20, 200, 1))
        else:
            input_lid = Input(shape=(20, 200, 10))
    elif(LIDAR_TYPE=='ABSOLUTE_LARGE'):
        if(FLATTENED):
            input_lid = Input(shape=(60, 330, 1))
        else:
            input_lid = Input(shape=(60, 330, 10))
    noisy_input_lid=GaussianNoise(0.01)(input_lid)
    layer = Conv2D(5, kernel_size=(5, 5), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(noisy_input_lid)
    layer = Conv2D(5, kernel_size=(5, 5), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(5, kernel_size=(5, 5), strides=2, activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(5, kernel_size=(5, 5), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(5, kernel_size=(5, 5), strides=2, activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(1, kernel_size=(3, 3), strides=(1, 2), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Flatten()(layer)
    out_lid = Dense(16, activation='relu')(layer)
    '''GPS branch'''
    input_coord = Input(shape=(3))
    noisy_input_coord=GaussianNoise(0.002)(input_coord)
    '''Concatenation'''
    concatenated = concatenate([out_lid, noisy_input_coord])
    reg_val = 0
    layer = Dense(64, activation='relu')(concatenated)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(64, activation='relu')(layer)
    predictions = Dense(256, activation='softmax')(layer)
    architecture = Model(inputs=[input_lid, input_coord], outputs=predictions)
    return architecture
