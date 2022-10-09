import os
import tensorflow as tf
import numpy as np
import h5py
from baseline_base_model import build_res_unet, get_mask

# Domain dimensions
simx, simy, simz = 128, 256, 30

# Assign GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0" #1,2,3

top_train_mean = 0
bot_train_mean = 0
cur_ts = 33

def seg_mask(y, eps=0):
    """
    Create two binary masks for water and CO2,
    separated by channel.
    """
    return tf.cast(tf.concat([y>eps, y<=eps], axis=-1), tf.float32)

def basic_mse(y_true, y_pred):    
    # Double pred
    actual_top = y_true[:,:,:,0:1]
    actual_bot = y_true[:,:,:,1:2]
    
    # MSE loss
    top_loss = ((actual_top - y_pred[:,:,:,0:1])**2)
    bot_loss = ((actual_bot - y_pred[:,:,:,1:2])**2)
    loss = (tf.math.reduce_mean(top_loss) + tf.math.reduce_mean(bot_loss)) / 2
    return loss

def custom_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()

    # Double pred
    actual_top = y_true[:,:,:,0:1]
    actual_bot = y_true[:,:,:,1:2]
    mask = y_true[:,:,:,2:3]

    # Crossentropy loss
    actual_mask = seg_mask(actual_top)
    pred_top_mask = seg_mask(y_pred[:,:,:,0:1], tf.math.reduce_min(actual_top[actual_top>0]))
    pred_bot_mask = seg_mask(y_pred[:,:,:,1:2], tf.math.reduce_min(actual_bot[actual_bot>0]))
    ce_loss = (bce(actual_mask, pred_top_mask) + bce(actual_mask, pred_bot_mask)) / 2

    # MSE loss
    top_loss = ((actual_top - y_pred[:,:,:,0:1])**2) * (tf.ones_like(y_pred[:,:,:,0:1])+mask_weight*mask)
    top_loss = tf.divide(tf.cast(top_loss, tf.float64), tf.constant(top_train_mean))
    bot_loss = ((actual_bot - y_pred[:,:,:,1:2])**2) * (tf.ones_like(y_pred[:,:,:,1:2])+mask_weight*mask)
    bot_loss = tf.divide(tf.cast(bot_loss, tf.float64), tf.constant(bot_train_mean))
    loss = (tf.math.reduce_mean(top_loss) + tf.math.reduce_mean(bot_loss)) / 2
    return loss + tf.cast(ce_loss, tf.float64)
  
# Load Data & Normalize
load_str = 'in/'
# fracs = ['1NW', '1fNW', '1sNW', '2NW', '2fNW', '2sNW','3NW', '3fNW', '3sNW', '4NW', '4fNW', '4sNW']
fracs = ['1NW', '1fNW', '1sNW', '2fNW', '2sNW','3NW', '3fNW', '4NW', '4fNW', '4sNW']
top_dry = np.array([np.load(load_str + 'top_' + x + '_' + str(cur_ts) + '.npy') for x in fracs])
bot_dry = np.array([np.load(load_str + 'bot_' + x + '_' + str(cur_ts) + '.npy') for x in fracs])
topf = np.array([np.load(load_str + 'topf_' + x + '_' + str(cur_ts) + '.npy') for x in fracs])
botf = np.array([np.load(load_str + 'botf_' + x + '_' + str(cur_ts) + '.npy') for x in fracs])
topf = np.array([np.maximum((topf[i] - bot_dry[i]), 0) / (top_dry[i] - bot_dry[i]) for i in range(len(fracs))])
botf = np.array([np.maximum((botf[i] - bot_dry[i]), 0) / (top_dry[i] - bot_dry[i]) for i in range(len(fracs))])

top_dry = np.array([top_dry[i]/simz for i in range(len(fracs))])
bot_dry = np.array([bot_dry[i]/simz for i in range(len(fracs))])

# # Augment across the x axis - XXX ONLY FINAL
# top_dry = np.concatenate([top_dry, top_dry[:,::-1,:]], axis=0)
# bot_dry = np.concatenate([bot_dry , bot_dry [:,::-1,:]], axis=0)
# topf = np.concatenate([topf, topf[:,::-1,:]], axis=0)
# botf = np.concatenate([botf, botf[:,::-1,:]], axis=0)

# # XXX DISTANCE FEATURE
# dist_from_inlet = np.full(top_dry.shape, np.squeeze(np.load('model_data/dist_from_inlet.npy'))[:,::-1])

# # Augment across the y axis, reverse distance input - XXX ONLY FINAL
# top_dry_rev = top_dry[:,:,::-1]
# bot_dry_rev = bot_dry[:,:,::-1]
# topf_rev = topf[:,:,::-1]
# botf_rev = botf[:,:,::-1]

# dist_from_inlet_rev = np.full(top_dry_rev.shape, np.squeeze(np.load('model_data/dist_from_inlet.npy')))

# # Set up training and testing data
# top_dry = np.concatenate([top_dry, top_dry_rev], axis=0)
# bot_dry = np.concatenate([bot_dry, bot_dry_rev], axis=0)
# topf = np.concatenate([topf, topf_rev], axis=0)
# botf = np.concatenate([botf, botf_rev], axis=0)
# dist_from_inlet = np.concatenate([dist_from_inlet, dist_from_inlet_rev], axis=0)

X = np.stack([top_dry, bot_dry], axis=-1)
# XXX DISTANCE FEATURE
# X = np.stack([top_dry, dist_from_inlet, bot_dry, dist_from_inlet], axis=-1)
y = np.stack([topf, botf], axis=-1)

# # XXX MASKS - ONLY FINAL
# masks = np.empty([y.shape[0], y.shape[1], y.shape[2], 1])

# for i in range(len(masks)):
#     masks[i] = get_mask(y[i,:,:,0])

# y = np.concatenate((y, masks), axis=-1)

# # Fix train/val/test sets XXX ONLY FINAL
# test_set = [3,15,27,39,8,20,32,44]

# X_test = X[test_set]; y_test = y[test_set]

# bool_inds = [x not in test_set for x in range(len(X))]
# X_train = X[bool_inds]; y_train = y[bool_inds]

# # XXX ONLY FINAL - Calculate mean of training set to scale top/bot loss.
# top_train_mean = np.mean(y_train[:,:,:,0], axis=0)
# top_train_mean = top_train_mean[None,:,:,None]
# top_train_mean[top_train_mean==0] = 1
# bot_train_mean = np.mean(y_train[:,:,:,1], axis=0)
# bot_train_mean = bot_train_mean[None,:,:,None]
# bot_train_mean[bot_train_mean==0] = 1

# Build and train model
tf.keras.backend.clear_session()

model = build_res_unet((128, 256, 2), (128, 256, 2), filters=32)
# XXX ONLY BASE model = build_res_unet((128, 256, 4), (128, 256, 2), filters=32)

mask_weight=10

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=basic_mse)
# XXX ONLY BASE model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=custom_loss)

model_name = 'base' + str(cur_ts)
save_filepath = 'out/' + model_name + '.h5'

check_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_filepath, monitor='loss', save_best_only=True, mode='min', save_weights_only=False
)

model.fit(X, y, epochs=16000, callbacks=[check_callback],
            batch_size=2, shuffle=True)
