import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pdb
import uuid
#from phase_pick_generator import get_generator

def build_model(batch_size=32):

    input_layer=tf.keras.layers.Input(batch_shape=(batch_size,None,5))
    lstm_f=tf.keras.layers.LSTM(100,return_sequences=True,stateful=True,recurrent_dropout=0.05)(input_layer)
    lstm_f=tf.keras.layers.LSTM(100,return_sequences=True,stateful=True)(lstm_f)
    out_head=tf.keras.layers.Dense(50)(lstm_f)
    out_head=tf.keras.layers.LeakyReLU()(out_head)
    out_head=tf.keras.layers.Dense(50)(out_head)
    out_head=tf.keras.layers.LeakyReLU()(out_head)
    out_head=tf.keras.layers.Dense(50)(out_head)
    out_head=tf.keras.layers.LeakyReLU()(out_head)
    regression=tf.keras.layers.Dense(5)(out_head)
    classification=tf.keras.layers.Dense(1,activation='sigmoid')(out_head)


    model =tf.keras.models.Model(input_layer,[regression,classification])
    opt=tf.keras.optimizers.Adam(1e-3)
    model.compile(loss=['mse','binary_crossentropy'],optimizer=opt)
    return model