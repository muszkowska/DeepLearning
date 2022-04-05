# DeepLearning

## Image classification with convolutional neural networks

Authors: Aleksandra Muszkowska, Mateusz Jastrzębiowski

### 1 Notebook worth to revise
* notebooks/results.ipynb


### 2 Network architectures
In our project, we will compare different neural network architectures to classify images from the
CIFAR-10 dataset.
* As the first architecture, we will implement a simple convolutional network – using the PyTorch
library.
*  Secend architecture will be expanded simple convolutional network with more layers.
* Next, we will use a pre-trained network (AlexNet, ResNet, .. or others). We will try two approaches:
  * Use the network as a feature extractor – freeze the weights for the entire network and train
only the last layer - the classifier.
  *  Use the first n layers – without changing the weights. Train the model on the last few layers.

### 3 Hyper-parameters tuning
We will investigate influence of the following hyper-parameter change on obtained results:
* Hyper-parameters related to training process: Learning rate, Number of epochs, Batch size.
* Hyper-parameters related to regularization: Dropout rate, Weight decay.
### 4 Data augmentation
We will test both standard and more sophisticated data augmentation techniques and choose those
which give the best results


