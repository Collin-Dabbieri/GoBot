# Some notes about alphago's RL policy network

# The strong policy network is a 13-layer convolutional network.
# All of these layers produce 19 × 19 filters;
# you consistently keep the original board size across the whole network.
# For this to work, you need to pad the inputs accordingly
# The first convolutional layer has a kernel size of 5,
# and all following layers work with a kernel size of 3.
# The last layer uses softmax activations and has one output filter,
# and the first 12 layers use ReLU activations and have 192 output filters each.

import numpy as np
import pickle
import os
import sys
import tensorflow as tf
backend='tensorflow'
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import math

from train_generator_class import DataGenerator as DataGeneratorTrain

########################## USER SPECIFIED VALUES ###############################################

conv=[50,50,50,50,50,50,50,50,50,50]        # list with number of filters in each convolutional layer (does not include output layer)
epochs=10                                   # number of training epochs
activation='elu'                            # activation function for convolutional filters
l2=0.001                                    # l2 regularization value
kernel_initializer='glorot_uniform'         # kernel initializer
bias_initializer='zeros'                    # bias initializer
padding='same'                              # padding for convolutional layers
output_activation='softmax'                 # activation function for output convolutional layer
verbose=1                                   # verbosity
kernel_size=[5,3,3,3,3,3,3,3,3,3]           # list of kernel sizes for each convolutional layer
loss='binary_crossentropy'                  # loss function for training
opt='adam'                                  # optimizer for training (adam or rmsprop)
lrate=0.0001                                # learning rate for training
shuffle=True                                # boolean for shuffling training indexes each epoch

files_dir='/home/pi/Go_Database/Processed/' # path to directory where training data, validation data, and partition file are located
filename_partition='partition.npy'          # filename of partition file
results_dir='/home/pi/repos/GoBot/results/' # path to directory for saving results

########################## FUNCTIONS ###########################################################

def build_model(args):
    '''
    Builds the CNN architecture
    params:
        args - dictionary containing all gridsearch hyperparams
    returns:
        model - tensorflow model object
    '''
    num_layers=len(args['conv'])

    # build input convolutional layer
    inpt = Input(shape=(19,19,2),name='input') # 2 filter input Go board
    x1=Conv2D(filters=args['conv'][0],
              kernel_size=args['kernel_size'][0],
              padding=args['padding'],
              activation=args['activation'],
              kernel_regularizer=tf.keras.regularizers.l2(args['l2']))(inpt)
    # build all subsequent convolutional layers
    for i in range(1,num_layers):
        x1=Conv2D(filters=args['conv'][i],
                  kernel_size=args['kernel_size'][i],
                  padding=args['padding'],
                  activation=args['activation'],
                  kernel_initializer=args['kernel_initializer'],
                  use_bias=True,
                  bias_initializer=args['bias_initializer'],
                  kernel_regularizer=tf.keras.regularizers.l2(args['l2']))(x1)

    # build output convolutional layer
    output=Conv2D(filters=1,
                  name='output',
                  kernel_size=3,
                  padding='same',
                  activation=args['output_activation'],
                  use_bias=True,
                  bias_initializer=args['bias_initializer'],
                  kernel_initializer=args['kernel_initializer'])(x1)

    model=Model(inputs=inpt,outputs=output)

    if args['opt']=="adam":
        opt=tf.keras.optimizers.Adam(lr=args['lrate'],beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)
    elif args['opt']=="rmsprop":
        opt="rmsprop"

    model.compile(loss=args['loss'], optimizer=opt, metrics=['categorical_accuracy'])

    if args['verbose']>0.:
        print(model.summary())

    return model

def generate_fname(args):
    '''
    This takes important model arguments and uses them to generate a filename base for the saved trained model and output pickle files

    params:
    args - dictionary object with all model hyperparameters
    returns:
    fbase - base filename to be used for trained model and output pickle file
    '''

    #here we want the filename to (ideally) express all of the information that might be changed from one experiment to the next
    conv_str='conv_'
    for i in args['conv']:
        conv_str=conv_str+str(i).zfill(2)+'_'
    l2_str='l2_'+str(args['l2']).zfill(6)+'_'
    lrate_str='lrate_'+str(args['lrate']).zfill(6)
    fbase=conv_str+l2_str+lrate_str
    return(fbase)

########################## MAIN LOOP ###########################################################


if __name__=='__main__':
    # build args from values in USER SPECIFIED VALUES
    args={}
    args['conv']=conv
    args['epochs']=epochs
    args['activation']=activation
    args['l2']=l2
    args['kernel_initializer']=kernel_initializer
    args['bias_initializer']=bias_initializer
    args['padding']=padding
    args['output_activation']=output_activation
    args['verbose']=verbose
    args['kernel_size']=kernel_size
    args['loss']=loss
    args['opt']=opt
    args['lrate']=lrate
    args['shuffle']=shuffle

    if args['verbose']>0:
        print("Building partition")

    partition=np.load(files_dir+filename_partition,allow_pickle=True)
    train_names=partition.item().get('train')
    validation_names=partition.item().get('validation')
    size_train=len(train_names)
    size_validation=len(validation_names)

    #load in generators for training and validation
    training_generator = DataGeneratorTrain(filenames=train_names,
                                            shuffle=args['shuffle'])
    validation_generator = DataGeneratorTrain(filenames=validation_names,
                                              shuffle=False) #there's no need to shuffle validation indexes

    if args['verbose']>0:
        print("Building model")

    model=build_model(args)
    num_params=model.count_params()

    if args['verbose']>0:
        print("Training model")

    fbase=generate_fname(args)

    #train over synthetic training data
    history=model.fit_generator(generator=training_generator,
                                validation_data=validation_generator,
                                epochs=args['epochs'],
                                use_multiprocessing=False, # can't overcome the error where memory fills up when using multiprocessing
                                verbose=args['verbose'])

    if args['verbose']>0:
        print("Writing Results")

    results={}
    results['args']=args
    results['history']=history.history
    results['num_params']=num_params

    #save results
    results['fname_base']=fbase
    fname="{:s}{:s}_results.npy".format(results_dir,fbase)
    np.save(fname,results)

    #save model
    model.save("{:s}{:s}_model.hdf5".format(results_dir,fbase))
