import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pdb
import uuid
#from phase_pick_generator import get_generator

def add_time(layers):
    input_layer,output_layer=layers

    output_slices=tf.split(output_layer,5,axis=-1)
    print(len(output_slices),output_slices[0])
    new_time=output_slices[-1]+input_layer[:,:,-1:]
    print(new_time,input_layer[:,:,-1:])
    new_output_list=output_slices[0:4]+[new_time]
    print(new_output_list)
    
    output=tf.concat(output_slices[0:4]+[new_time],axis=2)
    print(output)
    return output
    



def build_model(batch_size=32,use_residual=False):

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
    if use_residual:
        regression=tf.keras.layers.Lambda(add_time)([input_layer,regression])


    
    classification=tf.keras.layers.Dense(1,activation='sigmoid')(out_head)


    
    model =tf.keras.models.Model(input_layer,[regression,classification])
    opt=tf.keras.optimizers.Adam(1e-3)
    model.compile(loss=['mse','binary_crossentropy'],optimizer=opt)
    return model

if __name__=='__main__':
   m= build_model(use_residual=True)
   m.summary()
   m= build_model(use_residual=False)
   m.summary()
