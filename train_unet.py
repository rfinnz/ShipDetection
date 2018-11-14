#Loads training set of preprocessed downsampled images. Trains UNet model. Predicts and saves segmentations for all images


from __future__ import print_function
import os
import numpy as np
from PIL import Image
from keras import regularizers
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, Add, BatchNormalization
from keras.optimizers import Adam, Adadelta, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from os import rename, listdir
import matplotlib.pyplot as plt

# Dice metric for accuracy and loss
def dice_coef(y_true, y_pred):
    smooth = 0. #originally 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Dice loss
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
    
# Jaccard metric for accuracy and loss
def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return intersection / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection)

def jaccard_coef_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)

# Parameters
training_data_dir = '../Data/Test/'
gt_dir = '../Data/Test/'
data_dir = '../Data/'
pred_dir = '../Predictions/'
model_dir = '../Models/'
log_dir = '../Logs/'
option = 'batchnorm-lowreg_nostandardize' #only for record keeping, no effect on training
save_imgs = True
save_model = True
use_best_model = True #for predictions, load model saved as 'best' by lowest validation loss

block = 512
pad = 32
num_channels = 1
num_classes = 1

feature_scale = 0.2 #for scaling number of filters to use in convolution
dropout = 0.25
kernel = (3, 3)
batchnorm_momentum = 0.6
num_epochs = 2
batch_size = 1

num_images = 100
input_shape = (576, 1024, 1)

activation_hidden = 'relu'
padding = 'same'
kernel_reg = None#regularizers.l2(0.001)
opt_function = Adam()
kernel_init = 'glorot_uniform' #default is 'glorot_uniform'
loss_function = dice_coef_loss
metric_function = dice_coef
loss_function_str = 'dice_coef_loss' #used for reloading models, should match loss_function
metric_function_str = 'dice_coef'

# Derived parameters
jobid = '' #set later

def get_down_block(out, num_features):  
    out = Conv2D(num_features, kernel, activation=activation_hidden, padding=padding, kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(out)
    out = BatchNormalization(momentum=batchnorm_momentum)(out) 
    #out = Dropout(dropout)(out) #optional dropout  
    out = Conv2D(num_features, kernel, activation=activation_hidden, padding=padding, kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(out)   
    out = BatchNormalization(momentum=batchnorm_momentum)(out)
    #out = Dropout(dropout)(out) #optional dropout
    conv = out
    out = MaxPooling2D(pool_size=(2, 2))(out)
    
    return (out, conv)
    
def get_up_block(out, skip_conn, num_features):
    out = Conv2DTranspose(num_features, (2, 2), strides=(2, 2), padding=padding)(out)
    out = BatchNormalization()(out)
    out = concatenate([out, skip_conn], axis=3)    
    out = Conv2D(num_features, kernel, activation=activation_hidden, padding=padding, kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(out)
    out = BatchNormalization()(out)
    #out = Dropout(dropout)(out) #optional dropout 
    out = Conv2D(num_features, kernel, activation=activation_hidden, padding=padding, kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(out)
    out = BatchNormalization()(out)
    #out = Dropout(dropout)(out) #optional dropout
    conv = out
    
    return (out, conv)

# UNET model
def get_unet():
    inputs = Input(input_shape)  
     
    (out, conv1) = get_down_block(inputs, int(32*feature_scale))
    (out, conv2) = get_down_block(out, int(64*feature_scale))
    (out, conv3) = get_down_block(out, int(128*feature_scale))
    (out, conv4) = get_down_block(out, int(256*feature_scale))
    (out, conv5) = get_down_block(out, int(512*feature_scale))
    
    out = Conv2D(int(512*feature_scale), kernel, activation=activation_hidden, padding=padding, kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(out)
    out = BatchNormalization(momentum=batchnorm_momentum)(out)
    out = Conv2D(int(512*feature_scale), kernel, activation=activation_hidden, padding=padding, kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)(out)
    out = BatchNormalization(momentum=batchnorm_momentum)(out) 
    
    (out, conv6) = get_up_block(out, conv5, int(512*feature_scale))
    (out, conv7) = get_up_block(out, conv4, int(256*feature_scale))
    (out, conv8) = get_up_block(out, conv3, int(128*feature_scale))
    (out, conv9) = get_up_block(out, conv2, int(64*feature_scale))
    (out, conv10) = get_up_block(out, conv1, int(32*feature_scale))
    
    out = Conv2D(num_classes, (1, 1), activation='sigmoid')(out)
    model = Model(inputs=[inputs], outputs=[out])

    return model

#Record keeping function
def print_parameters():
    print('Job ID: ' + jobid)
    print('Option: ' + option)
    print('Training data directory: ' + training_data_dir)
    print('Ground truth directory: ' + gt_dir)
    print('Predictions directory: ' + pred_dir)
    print('Block: ' + str(block))
    print('Pad: ' + str(pad))
    print('Num channels: ' + str(num_channels))
    print('Num epochs: ' + str(num_epochs))
    print('Batch size: ' + str(batch_size))
    print('Kernel shape: ' + str(kernel))
    print('Dropout: ' + str(dropout))
    print('Feature scale: ' + str(feature_scale))
    print('Batchnormalization momentum: ' + str(batchnorm_momentum))
    print('Dropout: ' + str(dropout))
    print('Activation hidden: ' + activation_hidden)
    print('Kernel init: ' + kernel_init)
    if kernel_reg == None:
        print('Kernel regularization: None')
    else:
        print('Kernel regularization: ' + str(kernel_reg.get_config()))
    print('Optimzier: ' + str(opt_function))
     
     
def dice_coef_image(pred, target):
    dice = pred * target
    dice = np.sum(dice) * 2 * 100
    dice = dice / (np.sum(pred) + np.sum(target))
    return dice
    
#Reconstruct full image back from blocks. Remove context pad (32 pixels around each block) and block pad (around bottom and right sides) 
def reconstruct_from_blocks(y_pred, PID):
    img_size = np.asarray(Image.open(gt_dir + PID + "_GT.png")).shape #actual image size
    block_width, block_height = img_size
    block_width = int(np.ceil(block_width / float(block)))
    block_height = int(np.ceil(block_height / float(block)))
    
    full_img_padded = np.zeros((block_width*block, block_height*block, num_classes)) #image size w/ padding

    #remove context pad from each block   
    y_pred = y_pred[:, pad:-pad, pad:-pad, :]
    
    for j in range(y_pred.shape[0]):
        x = int(np.floor(j / block_height))
        y = np.mod(j, block_height)
        full_img_padded[x*block : (x+1)*block, y*block : (y+1)*block, :] = y_pred[j]
   
    full_img = full_img_padded[:img_size[0], :img_size[1], :] #remove block padding for entire image
    return full_img  
    
    
def load_training_data():
    x_img = plt.imread(training_data_dir + 'ir_frame3.jpg')
    x_img = x_img[np.newaxis, :, :, np.newaxis, 0]
    y_img = plt.imread(training_data_dir + 'ir_frame3_mask.png')
    y_img = y_img[np.newaxis, :, :, np.newaxis]
    
    x_train = np.repeat(x_img, num_images, axis=0)
    y_train = np.repeat(y_img, num_images, axis=0)
     
    return x_train, y_train
    

def train_and_predict():
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()    
    model.compile(optimizer=opt_function, loss=loss_function, metrics=[metric_function])  
    model_checkpoint = ModelCheckpoint(model_dir + jobid + '-' + option + '-best.h5', monitor='val_loss', save_best_only=True)
    
    print('-'*30)
    print('Model parameters')
    print('-'*30)
    print_parameters()
    model.summary()
    
    print('-'*30)
    print('Loading and preprocessing training data from: ' + training_data_dir)
    print('-'*30)
    x_train, y_train = load_training_data()
    x_train = x_train.astype('float32')
    train_max_val = np.amax(x_train)
    x_train /= train_max_val
    y_train = y_train.astype('float32')
     
    print('-'*30)
    print('Fitting model...')
    print('-'*30)          
    if save_model:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint])
        model.save(model_dir + jobid + '-' + option + '.h5')
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, shuffle=True, validation_split=0.2)   
    
    
    print('-'*30)
    print('Predict and save for all images...')
    print('-'*30)
    x_train = x_train[0]
    x_train = x_train[np.newaxis, :, :, :]      
    y_pred = model.predict(x_train, batch_size=1, verbose=2) 
    dice_score = dice_coef_image(y_train[0,:,:,0], y_pred[0,:,:,0])     
    print('Dice: ' + format(dice_score, '.4g'))
    
    Image.fromarray(((y_pred[0,:,:,0]>0.5)*255).astype('uint8')).save(pred_dir + "ir_frame3_pred.png") 
    del x_train, y_train #training data unused after this point, delete for memory savings 
    print('Finished.') 
    


if __name__ == '__main__':
    train_and_predict()


