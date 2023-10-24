import tensorflow as tf 
import keras
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Input
from tensorflow.keras.layers import Softmax,Flatten,Dense
from tensorflow.keras import layers
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model

def _e1x1(y,filters):
    y=Conv2D(filters,kernel_size=1,activation="relu")(y)
    return y

def _e3x3(y,filters):
    y=Conv2D(filters,kernel_size=3,activation='relu')(y)
    return y

def _maxpool2d(y):
    y=MaxPooling2D(pool_size=3,strides=2)(y)
    return y

def _squeeze(y,filters):
    """ the squeeze layer"""
    y=Conv2D(filters,kernel_size=1,activation='relu')(y)
    return y


def _expand(y,filters):
    """ The expand layer """
    #y=_e1x1(y,filters)
    #y=_e1x1(y,filters)
    y=_e1x1(y,filters)
    y=_e1x1(y,filters)

    #y=_e3x3(y,filters)
    #y=_e3x3(y,filters)
    y=_e3x3(y,filters)
    y=_e3x3(y,filters)
    return y

def _fire(y,filters,name):
    """ The fire module """
    y=_squeeze(y,filters)
    y=_expand(y,filters)
    return y


def Squeeze_net(input_shape):
    """
    The architecture of squeeze net using keras functionnal API
    """
    inputs=Input(shape=input_shape)
    y=Conv2D(filters=96,kernel_size=7,strides=2)(inputs)
    y=_maxpool2d(y)
    y=_fire(y,filters=128,name="fire2")
    y=_fire(y,filters=128,name="fire3")
    y=_fire(y,filters=256,name="fire4")
    y=_maxpool2d(y)
    y=_fire(y,filters=256,name="fire5")
    y=_fire(y,filters=384,name="fire6")
    y=_fire(y,filters=384,name="fire7")
    y=_maxpool2d(y)
    y=_fire(y,filters=512,name="fire8")
    y=_fire(y,filters=512,name="fire9")
    y=Dropout(.5)(y)
    y=Conv2D(filters=10,kernel_size=1,strides=1)(y)
    y=_maxpool2d(y)
    y=GlobalAveragePooling2D()(y)
    outputs=Dense(class_number)(y)
    model=keras.Model(inputs,outputs,name="Squeeze_net")
    return model

class_number=10 # just for test
input_shape=(400,400,2243) 
model=Squeeze_net(input_shape)
model.summary()
#plot_model(model,to_file='./rsc/squeeze_net.png',show_shapes=True)








