import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pdb
import uuid
from asso_utils import fix_x_batch,fix_y_batch_nomag_nodep

#from phase_pick_generator import get_generator


def lon_loss(y_true,y_pred):
   mask=tf.cast(tf.not_equal(y_true[:,:,-1] ,tf.zeros_like(y_true[:,:,-1])),'float32' )
   lonfac=1/0.0009280954987236569
   return tf.reduce_mean(tf.square(y_true[:,:,0]-y_pred[:,:,0])*mask)*lonfac

def lat_loss(y_true,y_pred):
    latfac=1/0.0008034709946970919
    mask=tf.cast(tf.not_equal(y_true[:,:,-1] ,tf.zeros_like(y_true[:,:,-1])),'float32' )
    return tf.reduce_mean(tf.square(y_true[:,:,1]-y_pred[:,:,1])*mask)*latfac

# def depth_loss(y_true,y_pred):
#     depfac=1/0.01
#     mask=tf.cast(tf.not_equal(y_true[:,:,-1] ,tf.zeros_like(y_true[:,:,-1])),'float32' )
#     return tf.reduce_mean(tf.square((y_true[:,:,2]-y_pred[:,:,2]))*mask)*depfac

# def mag_loss(y_true,y_pred):
 #   magfac=1/0.02
 #   mask=tf.cast(tf.not_equal(y_true[:,:,-1] ,tf.zeros_like(y_true[:,:,-1])),'float32' )
 #   return tf.reduce_mean(tf.square(y_true[:,:,3]-y_pred[:,:,3])*mask)*magfac

def time_loss(y_true,y_pred):
    timefac=1/0.005
    mask=tf.cast(tf.not_equal(y_true[:,:,-1] ,tf.zeros_like(y_true[:,:,-1])),'float32' )
    return tf.reduce_mean(tf.square(y_true[:,:,2]-y_pred[:,:,2])*mask)*timefac
    
def loss(y_true,y_pred):
    
    return (lon_loss(y_true,y_pred)
            +lat_loss(y_true,y_pred)
            +time_loss(y_true,y_pred))


class Checkpoint(tf.keras.callbacks.Callback):
    log_terms=['val_loss','loss','reg_lon_loss','reg_lat_loss','reg_time_loss','class_loss']

    def __init__(self,tag,restore=True):
        self.tag=tag
        self.restore=restore
        self.model_path='cp/'+tag+'_model.h5'

        if restore and os.path.exists('cp/'+tag+'_model.h5'):
            self.history,self.epoch,self.best_loss=pickle.load(open('cp/'+self.tag+'_history.pk','rb'))
        else:
            self.history={}
            for k in self.log_terms:
                self.history[k]=[]
            self.epoch=0
            self.best_loss=None
 
 

    def save(self):
        if not os.path.exists('cp'):os.mkdir('cp')
        self.model.save(self.model_path)
        pickle.dump([self.history,self.epoch,self.best_loss],open('cp/'+self.tag+'_history.pk','wb') )
        

    def on_epoch_end(self, epoch, logs=None):
        self.epoch+=1
        for term in self.log_terms:
            self.history[term].append(logs[term])

        if self.best_loss==None: self.best_loss=logs['loss']

        if logs['loss'] < self.best_loss:
            self.best_loss=logs['loss']
            self.model.save('cp/'+self.tag+'_best_model.h5')
        self.save()
            
def build_model(batch_size=None):

    input_layer=tf.keras.layers.Input(batch_shape=(batch_size,None,5))
    lstm_f=tf.keras.layers.LSTM(100,return_sequences=True,recurrent_dropout=0.05)(input_layer)
    lstm_b=tf.keras.layers.LSTM(100,return_sequences=True,recurrent_dropout=0.05,go_backwards=True)(input_layer)
    bidir=tf.keras.layers.Concatenate()([lstm_f,lstm_b])

    lstm_f=tf.keras.layers.LSTM(100,return_sequences=True)(bidir)
    lstm_b=tf.keras.layers.LSTM(100,return_sequences=True,recurrent_dropout=0.05, go_backwards=True)(bidir)
    bidir=tf.keras.layers.Concatenate()([lstm_f,lstm_b])

    out_head=tf.keras.layers.Dense(50)(bidir)
    out_head=tf.keras.layers.LeakyReLU()(out_head)
    out_head=tf.keras.layers.Dense(50)(out_head)
    out_head=tf.keras.layers.LeakyReLU()(out_head)
    out_head=tf.keras.layers.Dense(50)(out_head)
    out_head=tf.keras.layers.LeakyReLU()(out_head)

    regression=tf.keras.layers.Dense(3,name='reg')(out_head)

    classification=tf.keras.layers.Dense(1,activation='sigmoid',name='class')(out_head)

    model =tf.keras.models.Model(input_layer,[regression,classification])
    opt=tf.keras.optimizers.Adam(1e-3)
    model.compile(loss=[loss,'binary_crossentropy'],optimizer=opt, metrics=[[lon_loss,lat_loss,time_loss],[]])
    return model

if __name__=='__main__':
    
    datatype='small'

    all_data=np.load(datatype+'_comb_data.npy')
    train=np.load(datatype+'_train.npy') 
    test=np.load(datatype+'_test.npy')
    develop=np.load(datatype+'_develop.npy')
    
    x_train=all_data[train,:,:5]
    x_test=all_data[test,:,:5]
    x_develop=all_data[develop,:,:5]
    
    y_train=[all_data[train,:,5:-1],all_data[train,:,-1:]]
    y_test=[all_data[test,:,5:-1],all_data[test,:,-1:]]
    y_develop=[all_data[develop,:,5:-1],all_data[develop,:,-1:]]

    tag='v4'
    c=Checkpoint(tag=tag,restore=True)
    if os.path.exists(c.model_path):
       model= tf.keras.models.load_model('cp/'+tag+'_model.h5',
                                         custom_objects={'loss': loss,
                                                              'lon_loss':lon_loss,
                                                              'lat_loss':lat_loss,
                                                              'time_loss':time_loss}
                                           ) 
    else:
       model= build_model()


    xt=fix_x_batch(x_train.copy())
    y_rt,y_lt=fix_y_batch_nomag_nodep(*y_train.copy())
 
    xd=fix_x_batch(x_develop.copy())
    y_rd,y_ld=fix_y_batch_nomag_nodep(*y_develop.copy())

    m=model.fit(xt,[y_rt,y_lt],validation_data=(xd,[y_rd,y_ld]),
                epochs=150,
                callbacks=[c],
                batch_size=128)
