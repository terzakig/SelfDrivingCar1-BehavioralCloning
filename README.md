# Behavioral Cloning: Learning to Steer a Vehicle using Deep Learning on Video Sequences

## Overview
Train a deep neural network to autonomously steer a vehicle using visual input of the road from a camera. Generation of training data is done using a simulator which can be downloaded (linux version only) [here](https://1drv.ms/u/s!AtmapBHRVgqWgVogVyWCNGntVbNx). The trained model included in the repository works on the first track. To run the network, do the following:

1. Execute the Simulator and choose _autonomous mode_.
2. Run the model in a terminal window as follows: `python drive.py model.h5`.  

It is important to make sure you run **_keras_ with _tensorflow_** back-end. The model was trained in _tensorflow_ and will not have the same behavior when running with the _theano_ back-end. 

## Design
### Training 
File _model.py_ does the training of the network in _keras_. It uses a _data generator_ which parses local directories for training sequences and organizes them along with "putative" outputs in reasonably sized training batches for the network. Keras can handle training with data generators and this will make the overall process nicely. 
### The Network
The network uses convolutional layers at the bottom to train image features and thereafter utilizes _recurrent layers_ with [Long Short Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) neurons to "remember" _past features from the input images_. It also uses LSTMs to remember _past driving states_. I am conjecturing that adding memory to the network allowed me to train it with much less training data and without the need to generate additional images as Nvidia did [here](https://arxiv.org/abs/1604.07316). Although I haven't used this method to boost the traiing data, I am conjecturing that the network will perform great in both tracks if this method is employed. I will experiment as soon as I get the time...  
## Demo Video
A demo video of the vehicle steering autonomously for two laps of the track can be accessed [here](https://www.youtube.com/watch?v=OSy9ijPSalA). 
## Detailed Info
Details about how the network is structured and the rationale behind its conception can be found in yje [writeup_report.pdf](https://github.com/terzakig/SelfDrivingCar1-BehavioralCloning/blob/master/writeup_report.pdf) file included in this repository. 
