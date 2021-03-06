# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import keras
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Masking
from keras.layers import GlobalAveragePooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.regularizers import l2

def create_model_top_down(max_seq_len, n_features, n_domains=None, n_relations=None, 
                          units=128, drop_rate=0.5, rec_drop_rate=0.0, seed=None, batch_norm=False,
                          l2_reg=0.01, hidden_fc=0, mode=None, recurrent_type='LSTM'):
    mode = mode if mode else 'both_splitted'
    assert mode in ('both_splitted', 'relation', 'domain')
    
    recurrent_layers = dict(GRU=GRU, LSTM=LSTM)
    assert recurrent_type in recurrent_layers
    RecurrentLayer = recurrent_layers[recurrent_type]
    
    if seed is not None:
        np.random.seed(seed)
    
    x = input_features = Input(shape=(max_seq_len, n_features), name='input')
    
    x = Masking()(x)
    x = BatchNormalization()(x) if batch_norm else x

    x = RecurrentLayer(units, 
             bias_regularizer=l2(l2_reg),
             kernel_regularizer=l2(l2_reg),
             dropout=rec_drop_rate,
             recurrent_dropout=rec_drop_rate,
             unroll=True,
             name=recurrent_type)(x)

    x = BatchNormalization()(x) if batch_norm else x    
    
    x = Dropout(drop_rate)(x)
    
    for idx in range(hidden_fc):
        x = Dense(units,
                 activation='relu',
                 bias_regularizer=l2(l2_reg),
                 kernel_regularizer=l2(l2_reg),
                 name='fc' + str(idx))(x)

        x = BatchNormalization()(x) if batch_norm else x        
        
        x = Dropout(drop_rate)(x)
      
    if mode != 'relation':
        for idx in range(hidden_fc):
            x = Dense(units,
                     activation='relu',
                     bias_regularizer=l2(l2_reg),
                     kernel_regularizer=l2(l2_reg),
                     name='fc_d_' + str(idx))(x)

            x = BatchNormalization()(x) if batch_norm else x        
            x = Dropout(drop_rate)(x)
        
        domain = Dense(n_domains, name='domain',
                       activation='softmax',
                       bias_regularizer=l2(l2_reg),
                       kernel_regularizer=l2(l2_reg),
                      )(x)
        # extend the lstm output with the domain output
        x = keras.layers.concatenate([x, domain])

    if mode != 'domain':
        for idx in range(hidden_fc):
            x = Dense(units,
                     activation='relu',
                     bias_regularizer=l2(l2_reg),
                     kernel_regularizer=l2(l2_reg),
                     name='fc_r_' + str(idx))(x)

            x = BatchNormalization()(x) if batch_norm else x        
            x = Dropout(drop_rate)(x)
        
        relation = Dense(n_relations, name='relation',
                         activation='softmax',
                         bias_regularizer=l2(l2_reg),
                         kernel_regularizer=l2(l2_reg),
                        )(x)

    if mode == 'both_splitted':
        outputs = [domain, relation]
    elif mode == 'domain':
        outputs = [domain]
    else:
        outputs = [relation]
    
    model = Model(inputs=[input_features], outputs=outputs)
    return model

def create_model_bottom_up(max_seq_len, n_features, n_domains=None, n_relations=None, 
                          units=128, drop_rate=0.5, rec_drop_rate=0.0, seed=None, batch_norm=False,
                          l2_reg=0.01, hidden_fc=0, mode=None, recurrent_type='LSTM'):
    mode = mode if mode else 'both_splitted'
    assert mode in ('both_splitted', 'relation', 'domain')
    
    recurrent_layers = dict(GRU=GRU, LSTM=LSTM)
    assert recurrent_type in recurrent_layers
    RecurrentLayer = recurrent_layers[recurrent_type]

    if seed is not None:
        np.random.seed(seed)    
    
    x = input_features = Input(shape=(max_seq_len, n_features), name='input')
    x = Masking()(x)

    x = BatchNormalization()(x) if batch_norm else x
    
    x = RecurrentLayer(units,
             bias_regularizer=l2(l2_reg),
             kernel_regularizer=l2(l2_reg),
             dropout=rec_drop_rate,
             recurrent_dropout=rec_drop_rate,
             unroll=True,                       
             name=recurrent_type)(x)
   
    x = BatchNormalization()(x) if batch_norm else x

    x = Dropout(drop_rate, name='dropout_lstm')(x)
    
    for idx in range(hidden_fc):
        x = Dense(units, 
                 activation='relu',                  
                 bias_regularizer=l2(l2_reg),
                 kernel_regularizer=l2(l2_reg),
                 name='fc' + str(idx))(x)

        x = BatchNormalization()(x) if batch_norm else x
        x = Dropout(drop_rate)(x)
    
    # domain is a lineal combination of relation
    relation = Dense(n_relations, name='relation',
                     activation='softmax',
                     bias_regularizer=l2(l2_reg),
                     kernel_regularizer=l2(l2_reg),
                    )(x)
    
    if mode != 'relation':
        # use domain knowledge, weights are frozen
        domain = Dense(n_domains, name='domain',
                       activation='linear',
                       use_bias=False, trainable=True,
                      )(relation)

    if mode == 'both_splitted':
        outputs = [domain, relation]
    elif mode == 'domain':
        outputs = [domain]
    else:
        outputs = [relation]
    
    model = Model(inputs=[input_features], outputs=outputs)
    return model

def create_model_independent_outputs(
    max_seq_len, n_features, n_domains=None, n_relations=None, 
    units=128, drop_rate=0.5, rec_drop_rate=0.0, seed=None, batch_norm=False,
    l2_reg=0.01, hidden_fc=0, mode=None, recurrent_type='LSTM'):
    mode = mode if mode else 'both_splitted'
    assert mode in ('both_splitted', 'relation', 'domain')
    
    recurrent_layers = dict(GRU=GRU, LSTM=LSTM)
    assert recurrent_type in recurrent_layers
    RecurrentLayer = recurrent_layers[recurrent_type]

    if seed is not None:
        np.random.seed(seed)
        
    x = input_features = Input(shape=(max_seq_len, n_features), name='input')
    x = Masking()(x)
    x = BatchNormalization()(x) if batch_norm else x
    
    x = BatchNormalization()(x) if batch_norm else x
    
    x = RecurrentLayer(units, 
             bias_regularizer=l2(l2_reg),
             kernel_regularizer=l2(l2_reg),
             dropout=rec_drop_rate,
             recurrent_dropout=rec_drop_rate,
             unroll=True,
             name=recurrent_type)(x)
 
    x = BatchNormalization()(x) if batch_norm else x    
    shared_x = Dropout(drop_rate)(x)
    
    x = shared_x
    if mode != 'relation':
        for idx in range(hidden_fc):
            x = Dense(units,
                     activation='relu',
                     bias_regularizer=l2(l2_reg),
                     kernel_regularizer=l2(l2_reg),
                     name='fc_d_' + str(idx))(x)

            x = BatchNormalization()(x) if batch_norm else x        
            x = Dropout(drop_rate)(x)
            
        domain = Dense(n_domains, name='domain',
                       activation='softmax',
                       bias_regularizer=l2(l2_reg),
                       kernel_regularizer=l2(l2_reg),
                      )(x)

    x = shared_x        
    if mode != 'domain':
        for idx in range(hidden_fc):
            x = Dense(units,
                     activation='relu',
                     bias_regularizer=l2(l2_reg),
                     kernel_regularizer=l2(l2_reg),
                     name='fc_r_' + str(idx))(x)

            x = BatchNormalization()(x) if batch_norm else x        
            x = Dropout(drop_rate)(x)
            
        relation = Dense(n_relations, name='relation',
                         activation='softmax',
                         bias_regularizer=l2(l2_reg),
                         kernel_regularizer=l2(l2_reg),
                        )(x)

    if mode == 'both_splitted':
        outputs = [domain, relation]
    elif mode == 'domain':
        outputs = [domain]
    else:
        outputs = [relation]
    
    model = Model(inputs=[input_features], outputs=outputs)
    return model
