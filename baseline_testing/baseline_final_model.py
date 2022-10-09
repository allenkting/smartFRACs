import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, activations
from scipy.ndimage import distance_transform_edt as edist
import h5py
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D,  \
            BatchNormalization, LayerNormalization, Activation, Add, Multiply, \
            concatenate, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, \
            ConvLSTM2D

def pad_layer(x, padding_size=1):
    """
    Adds padding_size padding around x, periodic in the y-direction and repeating in
    the x-direction.
    """
    result = tf.concat([x[:,-padding_size:,:,:], x, x[:,:padding_size,:,:]], axis=1)
    left = result[:,:,:1,:]
    right= result[:,:,-1:,:]
    t = tf.constant([1,1,padding_size,1])
    result = tf.concat([tf.tile(left, t), result, tf.tile(right, t)], axis=2)
    
    return result

def get_mask(map):
    """
    Returns mask representing the interface.
    """
    dists = edist(map>0)
    dists_inv = edist(map<=0)
    total = ((dists>0) & (dists<=2)) + ((dists_inv>0) & (dists_inv<=2))
    return np.expand_dims(total, axis=-1)
"""
This file contains the necessary components to create fracnet
"""

def res_block(x, nb_filters, strides, groups=1):
    res_path = LayerNormalization()(x)
#     res_path = Activation(activation='gelu')(res_path) #MICRO
    res_path = pad_layer(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), strides=strides[0], groups=groups)(res_path)
#     res_path = LayerNormalization()(res_path) #MICRO
    res_path = Activation(activation='gelu')(res_path) 
    res_path = pad_layer(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), strides=strides[1], groups=groups)(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0], groups=groups)(x)
    shortcut = LayerNormalization()(shortcut)

    res_path = Add()([shortcut, res_path])
    return res_path


def encoder(x, filters):
    to_decoder = []

    main_path = pad_layer(x)
    main_path = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), groups=2)(main_path) # groups
    main_path = LayerNormalization()(main_path)
    main_path = Activation(activation='gelu')(main_path)

    main_path = pad_layer(main_path)
    main_path = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), groups=2)(main_path) # groups

    shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), groups=2)(x) # groups
    shortcut = LayerNormalization()(shortcut)

    main_path = Add()([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [filters*2, filters*2], [(2, 2), (1, 1)], groups=2) # groups
    to_decoder.append(main_path)
    
    main_path = res_block(main_path, [filters*4, filters*4], [(2, 2), (1, 1)], groups=2) # groups
    to_decoder.append(main_path) 

    return to_decoder


def decoder(x, from_encoder, filters):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [filters*4, filters*4], [(1, 1), (1, 1)], groups=2) # groups

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [filters*2, filters*2], [(1, 1), (1, 1)], groups=2) # groups

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [filters, filters], [(1, 1), (1, 1)], groups=2) # groups

    return main_path

def build_res_unet(input_shape, 
                   output_shape,
                   #mask_shape,
                   filters=16):
    
    inputs = Input(shape=input_shape)
    #masks = Input(shape=masks_shape)
    
    to_decoder = encoder(inputs, filters)

    path = res_block(to_decoder[2], [filters*8, filters*8], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder, filters=filters)
    
    path = Conv2D(filters=output_shape[2], kernel_size=(1, 1), groups=2)(path) # add groups
    
    #path = Multiply()([path,masks])
    
    # path = Activation(activation='sigmoid')(path)
    
    out_top = path[:,:,:,0:1]
    out_bot = path[:,:,:,1:2]
    out_top = tf.add(tf.keras.activations.relu(out_top), out_bot)

#     return tf.keras.models.Model(inputs=inputs, outputs=path)
    return tf.keras.models.Model(inputs=inputs, outputs=tf.concat([out_top, out_bot], axis=-1))