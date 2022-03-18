# DeepLearning

## Image classification with convolutional neural networks

Authors: Aleksandra Muszkowska, Mateusz Jastrzębiowski

### 1. Network architectures
In our project, we will compare different neural network architectures to classify images from the
CIFAR-10 dataset.
* As the first architecture, we will implement a simple convolutional network – using the PyTorch
library. Depending on the results and how much time it will take to train the network, we will
expand it with more layers.
* Next, we will use a pre-trained network (AlexNet, ResNet, .. or others). We will try two approaches:
  * Use the network as a feature extractor – freeze the weights for the entire network and train
only the last layer - the classifier.
  *  Use the first n layers – without changing the weights. Train the model on the last few layers.
*  As a third network – we will use other pre-trained convolutional network to compare results of
previous architectures with it.
### 2 Hyper-parameters tuning
We will investigate influence of the following hyper-parameter change on obtained results:
* Hyper-parameters related to training process: Learning rate, Number of epochs, Batch size.
* Hyper-parameters related to regularization: Dropout rate, Weight decay.
### 3 Data augmentation
We will test both standard and more sophisticated data augmentation techniques and choose those
which give the best results
