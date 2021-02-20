# Some notes about alphago's RL policy network

# The strong policy network is a 13-layer convolutional network.
# All of these layers produce 19 × 19 filters;
# you consistently keep the original board size across the whole network.
# For this to work, you need to pad the inputs accordingly
# The first convolutional layer has a kernel size of 5,
# and all following layers work with a kernel size of 3.
# The last layer uses softmax activations and has one output filter,
# and the first 12 layers use ReLU activations and have 192 output filters each.


# Each file in the training data contains one complete game of Go
# This generator will treat each game as a single batch
# So basically the size of each batch will vary each time

import numpy as np
import tensorflow as tf
import math

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,filenames,shuffle=True):
        'Initialization'
        self.filenames=filenames
        self.shuffle=shuffle #should be false on validation generator so model.predict will return correct indices
        self.indexes=np.arange(len(filenames))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # since each batch is a single game of Go, the number of batches is just the number of games of Go created
        return len(self.filenames)

    def on_epoch_end(self):
        'Updates the indexes after each epoch'
        self.indexes=np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self,index):
        'Generate one batch of data'

        #the generator will provide the index

        #Generate data
        ins,outs=self.__data_generation(self.filenames[index])

        return ins,outs

    def __data_generation(self,filename):
        'Generates data from one game'
        data=np.load(filename,allow_pickle=True)

        ins=data.item().get('ins') #shape num_moves,19,19,2
        outs=data.item().get('outs') #shape num_moves,19*19

        return ins,outs


