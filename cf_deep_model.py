''' 
@author: Mpountou
@year: 2020-2021
'''

from keras.models import Model
from keras.layers import Input, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Concatenate, Dense, Dropout
from keras.layers import Add, Activation, Lambda
from keras.layers import LeakyReLU

class cf_deep_model():

  def EmbeddingLayer(self,x,total_rows,total_cols):
    x = Embedding(total_rows, total_cols, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
    x = Reshape((total_cols,))(x)
    return x

  def __init__(self,t_users,t_items,factors,min_rating,max_rating):
    
    self.t_users = t_users
    self.t_items = t_items
    self.factors = factors

    user = Input(shape=(1,))
    u = self.EmbeddingLayer(user,t_users, factors)
    
    item = Input(shape=(1,))
    i = self.EmbeddingLayer(item,t_items, factors)
    
    x = Concatenate()([u, i])
    #x = Dropout(0.1)(x)
    
    x = Dense(4096, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.55)(x)

    x = Dense(2048, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.55)(x)

    x = Dense(1024, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(512, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(256, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(128, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    
    
    x = Dense(1, kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    #x = LeakyReLU(alpha=0.1)(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)    
    self.model = Model(inputs=[user, item], outputs=x)
    #opt = Adam(lr=0.01)
 
    #self.model.compile(loss='mean_absolute_error', optimizer=opt)
