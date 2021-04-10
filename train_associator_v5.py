import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pdb
import uuid
from asso_utils import fix_x_batch,fix_y_batch_phases

class Checkpoint(tf.keras.callbacks.Callback):
    log_terms=['loss','val_loss','class_loss','phase_loss']

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


def class_loss(y_true,y_pred):
    loss=tf.keras.losses.binary_crossentropy(
        y_true[:,:,0], y_pred[:,:,0], from_logits=False, label_smoothing=0
    )

    return tf.reduce_mean(loss)

def phase_loss(y_true,y_pred):
    mask=y_true[:,:,0]
    loss=tf.keras.losses.binary_crossentropy(
        y_true[:,:,1:], y_pred[:,:,1:], from_logits=False, label_smoothing=0
    )
    return tf.reduce_mean(loss*mask)

def loss(y_true,y_pred):
    loss=phase_loss(y_true,y_pred)+class_loss(y_true,y_pred)
    return loss
            
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

    classification=tf.keras.layers.Dense(2,activation='sigmoid',name='class')(out_head)

    model =tf.keras.models.Model(input_layer,[classification])
    opt=tf.keras.optimizers.Adam(1e-3)
    model.compile(loss=[loss],optimizer=opt, metrics=[class_loss, phase_loss])
    return model


if __name__=='__main__':
    
    datatype='phases'

    all_data=np.load(datatype+'_comb_data.npy')
    train=np.load(datatype+'_train.npy') 
    test=np.load(datatype+'_test.npy')
    develop=np.load(datatype+'_develop.npy')

    x_train=all_data[train,:,:5]
    x_test=all_data[test,:,:5]
    x_develop=all_data[develop,:,:5]

    # y values are P or S and EQ or no
    y_train=[all_data[train,:,5],all_data[train,:,6]]
    y_test=[all_data[test,:,5],all_data[test,:,6]]
    y_develop=[all_data[develop,:,5],all_data[develop,:,6]]

    tag='v3'
    c=Checkpoint(tag=tag,restore=True)
    if os.path.exists(c.model_path):
       model= tf.keras.models.load_model('cp/'+tag+'_model.h5',
                                         custom_objects={'loss':loss,'phase_loss':phase_loss,'class_loss':class_loss}) 
    else:
       model= build_model()
       
    print(model.summary)
    
    xt=fix_x_batch(x_train.copy())
    y_rt=fix_y_batch_phases(*y_train.copy())
 
    xd=fix_x_batch(x_develop.copy())
    y_rd=fix_y_batch_phases(*y_develop.copy())

    m=model.fit(xt,y_rt,validation_data=(xd,y_rd),
                epochs=100,
                callbacks=[c],
                batch_size=128)
