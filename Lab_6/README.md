# Training-Neural-Nets---Guidelines
This article focusses on the problems that I faced during training of a recurrent neural network. Few of the problems are general for all the neural networks. 

PS - Keras with TensorFlow backend was used to code the RNN
****

## What batch size should be used to train the network?
     model.fit(load_train, y_train, batch_size=2048, nb_epoch=1000, verbose=1 , validation_split = 0.10)
**Batch size is very small** - 
 1.  Training the neural network takes long time
 2.  The trained model may not be accurate enough
 3.  Training the network using small batch size requires less memory 
 4.  Training loss fluctuates a lot.
     As shown in image below : ![picture alt](http://cs231n.github.io/assets/nn3/loss.jpeg "Fluctuations in training loss") 

**Batch size is very large** - 
 1.  Training the neural network takes long time
 2.  Training the network using large batch size requires more memory and if memory(less RAM) is a constraint then batch size must be small

**** 
## Interpreting training and validation loss
     hist = model.fit(load_train, y_train, batch_size=2048, nb_epoch=1000, verbose=1 , validation_split = 0.10)
     y = hist.history
     plt.plot(x , y['loss'] , label = 'TRAINING LOSS')
     plt.xlabel('iterations')
     plt.hold(1)
     plt.plot(x , y['val_loss'] , label = 'VALIDATION LOSS')
     plt.legend(loc = 'upper left')
     plt.show()
     
**Training loss <<< Validation loss --> overfitting model**
 1. Model has learnt noise in the data or has learnt some features which ain't in the validation set. 
 2. Decrease your network size OR add dropout layer OR increase dropout value. 
 3. If training data and validation data belong to different distributions or have different features, increase or decrease the  validation split to have similar training and validation sets.
      
**Training loss ~ Validation loss --> underfitting model**
 1. Model has learnt features limited to the validation data only and hence performs better on the validation dataset.
 2. Increase the size of your model (either number of layers or the raw number of neurons per layer)
 3. Changing the validation set may help in some scenarios.

****
## How to deal with an overfitting model?
**Adding dropout layer**   Dropout layer with dropout value 0.5 works as follows : 
 ![picture alt](https://cdn-images-1.medium.com/max/800/1*IrdJ5PghD9YoOyVAQ73MJw.gif "Dropout Layer") 

**L2 and L1 regularisation**
 1. Weight regularization - will constantly decay the weights.
 2. Activity regularization - will tend to make the output of the layer smaller (used to regularize the output of a neural
 network.)
 
**Early Stopping**

          from keras.callbacks import EarlyStopping
          early_stopping = EarlyStopping(monitor = 'val_loss' , patience = 0.9)
          .
          .
          model.fit(load_train, y_train, batch_size=2048, nb_epoch=1000,validation_split = 0.10, callbacks = [early_stopping])

  **Patience in early stopping -** 
  Number of epochs to wait before early stop, if there is no progress on validation set. Patience is generally between 10-100
  with the range 10-20 being more common.

****
## Which activation functions to use?
**Hidden Layers -**
 1. Shallow Network - tanh or Sigmoid
 2. Deep Network - ReLU or Softplus 

**Output Layer -**
 1. Regression - Linear (makes no sense in hidden layers)
 2. Classification - Softmax (makes no sense in hidden layers)

****
## Which Optimizer to use?
     RMSprop is an extension of Adagrad that deals with its radically diminishing learning rates. It is identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numinator update rule. Adam, finally, adds bias-correction and momentum to RMSprop. Insofar, RMSprop, Adadelta, and Adam are very similar algorithms that do well in similar circumstances. Kingma et al. Bias-correction helps Adam slightly outperform RMSprop towards the end of optimization as gradients become sparser. Insofar, Adam might be the best overall choice.

****
## When is the learning rate too high or too low?
   ![picture alt](http://cs231n.github.io/assets/nn3/learningrates.jpeg "Fluctuations in training loss") 

       Learning rate can be changed in keras by importing the 'LearningRateScheduler' from keras.callbacks
       
****
## What does return_sequences = False/True do?
     model.add(LSTM( input_dim=1, output_dim = 16, return_sequences = True))

1.   return_sequences=False
     If we consider input shape for lstm layer (nb_samples, timesteps, input_dim) and number of neurons in this layer equals to  hidden_neurons, then the output shape will be (nb_samples,hidden_neurons) for this layer, which means we have only the last output for the whole sequence in each lstm neuron.

2.    return_sequences=True:
     If we consider input shape for lstm layer (nb_samples, timesteps, input_dim) and number of neurons in this layer equals to hidden_neurons, then the output shape will be (nb_samples, timesteps ,hidden_neurons) for this layer, which means we have the full sequence as output in each lstm neuron.

****
****       
       
   


     

 
   
    


