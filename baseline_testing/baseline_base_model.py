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
    total = ((dists>0) & (dists<=1)) + ((dists_inv>0) & (dists_inv<=1))
    return np.expand_dims(total, axis=-1)

def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = Add()([shortcut, res_path])
    return res_path

def encoder(x, filters):
    to_decoder = []

    main_path = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = Add()([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [filters*2, filters*2], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [filters*4, filters*4], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder, filters):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [filters*4, filters*4], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [filters*2, filters*2], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [filters, filters], [(1, 1), (1, 1)])

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
    
    path = Conv2D(filters=output_shape[2], kernel_size=(1, 1), padding="same")(path)
    
    return tf.keras.models.Model(inputs=inputs, outputs=path)
