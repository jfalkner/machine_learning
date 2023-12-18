# Text classification using LenNet-5 and the MNIST dataset

LeNet-5 is a well-known, relatively old and commonly used application of machine learning for teaching. More specifically, the architeture is called a Convolutional Neural Network (CNN), which can be used to classify images. Any labeled images can be used; however, the original paper used a set of hadwritten digits and produced an output of 10 values, each representing the likliehood that the image represented a given digit: 0-9.

Below is LeNet-5's Architecture as shown in the manuscript. Don't worry about understanding it all immediately -- it'll have a PyTorch implementation shortly. Focus on that the input is a 32x32 pixel image and the output is a 10 values (aka a 1x10 matrix)

![LeNet-5 Architecture](lenet-5_architecture.png)

## LeNet-5 and MNIST
The LeNet-5 neural network architecture was published in 1998. It is also featured as the example neural network for PyTorch's ["Introduction to PyTorch" video](https://youtu.be/IC0_FRiX-sw). A TensorFlow 2.0 implementation example is [here](https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342). More links to various articles and videos feature LeNet-5 are at the end of this write-up.

MNIST is a dataset of handwritten digits that is hosted by the National Institute of Standards and Technology. This is a great, freely available dataset that can be used to replicate training a classifier similar to the original LenNet-5 manuscript.

## Algorith in PyTorch

> WIP: ...

## More links

There are many more resources related to LeNet-5. Many are repeats of the same concept or implementations in different languages or trained with different data sets.  This section enumerates the links, grouping by related implementation and sorted roughly by the order I'd suggest reading them.

*Manuscripts and DataSets*

* ["Gradient-based learning applied to document recognition" by Y. Lecun et al, 1988](https://ieeexplore.ieee.org/document/726791) (note [Stanford hosts a copy of the PDF](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf))
* [MNIST](http://yann.lecun.com/exdb/mnist/) and [MNIST wiki article](https://en.wikipedia.org/wiki/MNIST_database)
  * See [Kaggle's copy of the original MNIST data](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data)
  * [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) was published in 2017

*PyTorch*

* ["Introduction to PyTorch" video](https://youtu.be/IC0_FRiX-sw)

*TensorFlow*

* [Understanding and Implementing LeNet-5 CNN Architecture](https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342)

*Others*

* [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data) copy and link to code that uses it
