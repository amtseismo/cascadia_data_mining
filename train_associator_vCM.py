import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pdb
import uuid
from asso_utils import fix_x_batch,fix_y_batch_phases
from asso_data import x_train,y_train,x_develop,y_develop

class Checkpoint(tf.keras.callbacks.Callback):
    log_terms=['val_cor_loss','cor_loss','val_dec_loss','dec_loss']

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
        y_true[:,:,1], y_pred[:,:,1], from_logits=False, label_smoothing=0
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
    
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(50,return_state=True,return_sequences=True)(bidir)
    

    decoder_inputs=tf.keras.layers.Input(batch_shape=(batch_size,None,1))

    decoder=tf.keras.layers.LSTM(50, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_inputs,
                                     initial_state=[state_h,state_c])
    

    correlation_matrix=tf.keras.layers.Dot(axes=2)([encoder_outputs,decoder_outputs])
    correlation_matrix=tf.keras.layers.Softmax(axis=2,name='cor')(correlation_matrix)

    
    c_out=tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=1,keepdims=True) )(correlation_matrix)
    c_out=tf.keras.layers.Permute((2,1))(c_out)

    decoder_outputs=tf.keras.layers.Concatenate()([c_out,decoder_outputs])

    decoder_outputs=tf.keras.layers.Dense(50)(decoder_outputs)
    decoder_outputs=tf.keras.layers.LeakyReLU()(decoder_outputs)
    decoder_outputs=tf.keras.layers.Dense(1,activation='sigmoid',name='dec')(decoder_outputs)
    
    
    

    

    model =tf.keras.models.Model([input_layer,decoder_inputs],[decoder_outputs,correlation_matrix])
    opt=tf.keras.optimizers.Adam(1e-3)
    model.compile(loss=['binary_crossentropy','sparse_categorical_crossentropy'],optimizer=opt)
    model.summary()

    


    return model

#Reshape data for training
def fix_data(x,y):
    new_x=fix_x_batch(x.copy())
    new_y=y.copy()
    
    #Renumber eq to be sequential
    for i,v in enumerate(y):
        evs=list(set(y[i,:,0]))
        for n,pick in enumerate(v):
            new_y[i,n,0]=evs.index(y[i,n,0])

    #Length here is the maximum number of eq+1
    length=int(np.max(evs)+1)

    #Create output sequence targets 1 for eq 0 otherwise
    neq=np.squeeze(np.max(new_y,axis=1))
    decoder_targets= np.zeros((len(y),length,1))             
    for b,v in enumerate(neq):
        decoder_targets[b,:int(v)+1,0]=1
    #Create decoder inputs, which will be just zeros always (but the number of zeros determins the number of decoder steps)
    decoder_inputs=np.zeros( (len(x),length,1) )

    output= ([new_x,decoder_inputs],[decoder_targets.astype('int32'),new_y])
    return output

if __name__=='__main__':
   tag='vCM'

   c=Checkpoint(tag=tag,restore=True)
   if os.path.exists(c.model_path):
      model= tf.keras.models.load_model('cp/'+tag+'_model.h5',
                                        custom_objects={}) 
   else:
      model= build_model()

   xt,yt=fix_data(x_train,y_train)
   xd,yd=fix_data(x_develop,y_develop)
 
   m=model.fit(xt,yt,validation_data=(xd,yd),
               epochs=1000,
               callbacks=[c],
               batch_size=128)
