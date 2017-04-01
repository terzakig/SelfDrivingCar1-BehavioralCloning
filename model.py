############## George Terzakis ############
##
## Self-driving model

# http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/

import os
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers.wrappers import TimeDistributed
from keras.engine.topology import Merge

from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

#from IPython.display import SVG
#from keras.utils.visualize_util import model_to_dot


import numpy as np
import cv2

###################  LOADING DATA HERE ############################

# My training data generator.
# It looks for directories named: "TRAIN_*" and 
# periodically delivers "num_frames" of training tuples 
# as ( < zoomed image by zoom>, <Steering Angle, Throttle, Break, Speed> )  
def data_generator(img_height,      # desired image height 
                   img_width,       # desired image width
                   data_prefix = 'TRAIN_',  # this prefix is 
                                           #used to locate the data directory.
                                           # Use "TRAIN_" or "VALIDATION_"
                   path = '/home/george/SelfDrivingCar/CarSimulator',
                   useSideViews = False, 
                   toGrayScale = False,
                   batch_size = 64):
              
         
    
    data_dirs = []
    
    for fname in os.listdir(path):
        if (fname[0:len(data_prefix)]==data_prefix):
            data_dirs.append(fname)
    # Now implementing the infinite loop that delivers the data
    currentDir = 0
    x_driving_state = []
    x_img_center = []
    x_img_left = []
    x_img_right = []
    y = []    
    # loop forever... 
    while 1:
        dirname = path+"/" + data_dirs[currentDir] + "/"
        f = open(dirname+"driving_log.csv")
        prev_driving_state = np.array([[0, 0, 0]])        
        for line in f:
            # split the log line entry 
            strs = line.split(',')
            # get the image file names
            img_name_center = strs[0]
            img_name_left = strs[1]
            img_name_right = strs[2]
                
            # get the output values
            steering_angle = np.float32(strs[3])
            throttle = np.float32(strs[4])
            #breaks = np.float32(strs[5])
            speed = np.float32(strs[6])
            
            # opening images and resizing to zoom * 100 %
            img_center = plt.imread(img_name_center)
            
            # input for only center view
            if not useSideViews:
                # resizing
                img_center = cv2.resize(img_center, (img_width, img_height))
                                
                if toGrayScale:
                    img_center = cv2.cvtColor(img_center, cv2.COLOR_RGB2GRAY)
                # finally, fill-in the tuple tensors 
                img_center = np.array(img_center - 128.0) / 128.0
                x_img_center.append(img_center)
            else: # making input with all three views
                # opening images and resizing to zoom * 100 %
                img_left = plt.imread(img_name_left)
                img_right = plt.imread(img_name_right)
                # resizing
                img_center = cv2.resize(img_center, (img_width, img_height))
                img_left = cv2.resize(img_left, (img_width, img_height))
                img_right = cv2.resize(img_right, (img_width, img_height))
                
                if toGrayScale:
                    img_center = cv2.cvtColor(img_center, cv2.COLOR_RGB2GRAY)
                    img_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)
                    img_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)
                img_center = np.array(img_center - 128.0) / 128.0
                img_left = np.array(img_left - 128.0) / 128.0
                img_right = np.array(img_right - 128.0) / 128.0
                # finally, assigning values for the tuple tensors
                # In the case of 3 images we just make a list of the inputs,
                # which will be fed to 3 distinct RNN feature models.
                # Thje 3 models will be evantually merged into a single layer 
                x_img_center.append(img_center)
                x_img_left.append(img_left)
                x_img_right.append(img_right)
            # append the driving state training input
            x_driving_state.append(prev_driving_state)
            # y is the same in any case
            y.append([steering_angle, throttle ])
            # update the previosu steering angle
            prev_driving_state = np.array([[steering_angle, throttle, speed]])
            # return the tuple if the batch is full
            if len(x_driving_state)==batch_size:
                if useSideViews:
                    train_data = ([np.array(x_driving_state), np.array(x_img_center), np.array(x_img_left), np.array(x_img_right)], np.array(y))
                else:
                    train_data = ([np.array(x_driving_state), np.array(x_img_center)], np.array(y))
                yield train_data
                x_driving_state = []
                x_img_right = []
                x_img_left = []
                x_img_center = []
                y = []
        # close the file and move on to some other training directory
        f.close()
        # circle around the training directories
        currentDir = (currentDir + 1) % len(data_dirs)






# Create an image feature memory branch (i.e., single input) RNN model
def FeatureRNNModel(img_height, 
                    img_width, 
                    toGrayScale = False,
                    conv_output_depth = 32, 
                    filter_height = 12, 
                    filter_width = 12, 
                    pool_height = 8,
                    pool_width = 8,
                    conv_stride = 3,
                    pool_stride = 2,
                    top_activation = 'relu'):
    
    # Convolution and max-pooling parameters 
    conv_strides = (conv_stride, conv_stride)   # convolution strides
    pool_strides = (pool_stride, pool_stride)   # pooling strides
    padding = 'valid'  # padding      
        
    
    model = Sequential()
    ###### Add the usual convolutional layer #########
    if toGrayScale:
        i_shape = (img_height, img_width, 1)
    else:
        i_shape = (img_height, img_width, 3)
    model.add(Convolution2D(conv_output_depth, 
                            filter_height, 
                            filter_width, 
                            subsample = conv_strides, 
                            border_mode=padding, 
                            input_shape=i_shape,
                            dim_ordering = 'tf'))
    
    ##### Get the convolution output shape for Valid padding :        
    assert(padding == 'valid' or padding == 'same')    
    if (padding == 'valid'):    
        conv_height = math.ceil(float(img_height - filter_height + 1) / float(conv_strides[0]))
        conv_width  = math.ceil(float(img_width - filter_width + 1) / float(conv_strides[1])) 
    elif (padding == 'same'):
        conv_height = math.ceil(float(img_height) / float(conv_strides[0]))
        conv_width  = math.ceil(float(img_width) / float(conv_strides[1])) 
        
    print("---------- 1. Convolutional Layer ----------------")
    print("Convolution output rows : ", conv_height)
    print("Convolution output columns : ", conv_width)
    print("Convolution output depth : ", conv_output_depth)
    print()
    
    ######### Now adding a max-pooling layer #############
    model.add(MaxPooling2D(pool_size = (pool_height, pool_width),
                           strides = pool_strides, 
                           border_mode = padding,
                           dim_ordering = 'tf') )
    model.add(Activation('relu'))
    if (padding == 'valid'):    
        mp_height = math.ceil(float(conv_height - pool_height + 1) / pool_strides[0])
        mp_width  = math.ceil(float(conv_width - pool_width + 1) / pool_strides[1]) 
    elif (padding == 'same'):
        mp_height = math.ceil(float(conv_height) / pool_strides[0])
        mp_width  = math.ceil(float(conv_width) / pool_strides[1]) 
    
    print("---------- 2. Max-pooling Layer ------------------")
    print("Pooling rows : ", mp_height)
    print("Pooling columns: ", mp_width)
    print("Pooling depth (same as convolution) : ", conv_output_depth)
    ###  Convolution output is : (samples, channels, rows, cols)
    ###                           where "samples" is the batch size 
    #If the ooutput_dim = 'th', then we need to permute the convolutional output before reshaping
    # in order to "send" the filter output to the last position of the quadruple
    #model.add(Permute((2, 3, 1), input_shape=(conv_output_depth, mp_height, mp_width)))
    
    # Now reshaping the wxhxcon_output_depth to (w*h)xcon_output_depth
    # and god speed......
    model.add(Reshape((np.int32(mp_width * mp_height), conv_output_depth)))    
    model.add(LSTM(160, 
                   input_shape=(1, conv_output_depth), 
                   activation = 'sigmoid', 
                   inner_activation='hard_sigmoid',
                   return_sequences = False))
    # dropouts on the LSTMS. Hope this is a good idea...
    #model.add(Dropout(p = 0.5))
    
    model.add(Dense(80, bias = True))
    model.add(Activation(top_activation))    
    # another dropout on the dense layer
    #model.add(Dropout(p = 0.5))
    #model.add(Dense(40, bias = True))
    #model.add(Activation('sigmoid')) 
#    model.add(Reshape((10, 5), input_shape = (60,)) )    
#    model.add(LSTM(5, input_shape = (1, 5)))
#    model.add(Dense(10, bias = True))
#    model.add(Activation('linear'))
#    
    #model.add(Dense(3, bias = True))
    #model.add(Activation(top_activation))    
    
    
    return model


# A second model on remembering driving 
def DrivingMemoryModel(nb_LSTM_Outputs = 8,
                       nb_Dense_Outputs = 30,
                       top_activation = 'relu'):
    # input structure: [<steering angle>, <throttle>, <breaks>, <speed>]    
    model = Sequential()
    # adding an LSTM directly on the input: 
    model.add(LSTM(nb_LSTM_Outputs, 
                   input_shape = (1, 3), 
                    activation = 'sigmoid', 
                    inner_activation = 'hard_sigmoid',
                    return_sequences = False))
    # Dropout on the LSTMs
    #model.add(Dropout(p = 0.5))
    # and the necessary dense layer...
    model.add(Dense(nb_Dense_Outputs, bias = True))
    model.add(Activation(top_activation))
    
    return model
    
    
    


########## CREATE, COMPILE AND TRAIN THE MODEL a la keras ###############
#
#width_ = 80
#height_ = 160
width_ = 160
height_ = 110
useGrayScale = False

image_model = FeatureRNNModel(img_height = height_, img_width = width_, toGrayScale = useGrayScale)
driving_model = DrivingMemoryModel()
model = Sequential()
model.add(Merge([driving_model, image_model], mode='concat', concat_axis=1))

model.add(Dense(30))
model.add(Activation('relu'))

#model.add(Dropout(p = 0.5))

model.add(Dense(20))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('linear'))

# dump the model in a png

# uncomment to load the weights and train over existing net
#model.load_weights('./model.h5')

model.compile(loss='mse', 
              optimizer='adam'
              #metrics=['mse']
             )
#plot(model, to_file='model.png')

model.fit_generator(data_generator(img_height = height_, 
                                   img_width = width_,
                                   toGrayScale = useGrayScale),
                    validation_data = data_generator(img_height = height_, 
                                                     img_width = width_, 
                                                     data_prefix = 'VALIDATION_',
                                                     toGrayScale = useGrayScale
                                                     ),
                    nb_val_samples = 1024,
                    samples_per_epoch = 2048, 
                    nb_epoch = 100)

# Saving the model
model.save('./model.h5')
