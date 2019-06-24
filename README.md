### Latin letter recognition by 2 ways: Convolutional Neural Network and Multilayer Perceptron

#### Dataset - [EMNIST Letters](https://www.nist.gov/node/1298471/emnist-dataset)

---
#### How to run: 
```python src/CNN.py```   
```python src/MLP.py```

---
#### Examples of input image:

![examples of input image](https://user-images.githubusercontent.com/42957722/59995239-08d21500-9670-11e9-892e-74cf806223ae.PNG)

### Examples of learning results
#### MLP
![MLP Results](https://user-images.githubusercontent.com/42957722/59995240-08d21500-9670-11e9-98ce-08355f3b51c1.PNG)
#### CNN
![CNN Results](https://user-images.githubusercontent.com/42957722/59995238-08d21500-9670-11e9-8d86-2ca77f7ea960.PNG)

## Information about each type of neural network

### Multi-layer perceptron
Multi-layered perceptrons are called direct propagation neural networks. The input signal in such networks spreads in the forward direction, from layer to layer. The multilayer perceptron in general representation consists of the following elements:<br>
- The set of input nodes that form the input layer;<br>
- One or more hidden layers of computational neurons;<br>
- Single output layer of neurons.<br>

properties:<br>
- Each neuron has a non-linear activation function.<br>
- Several hidden layers<br>
- High connectivity<br>

#### Architecture of MLP
![MLP Arch](https://user-images.githubusercontent.com/42957722/59995161-d32d2c00-966f-11e9-9241-02d85ce22f6c.png)

### Convolutional neural network
A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with a multiplication or other dot product. The activation function is commonly a RELU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution. The final convolution, in turn, often involves backpropagation in order to more accurately weight the end product.

Convolutional neural network in general representation consists of the following elements:<br>
- The set of input nodes that form the input layer;<br>
- Convolutional layers<br>
- Pooling layers<br>
- Fully connected layers<br>


Each neuron in a neural network computes an output value by applying a specific function to the input values coming from the receptive field in the previous layer. The function that is applied to the input values is determined by a vector of weights and a bias (typically real numbers). Learning, in a neural network, progresses by making iterative adjustments to these biases and weights.

The vector of weights and the bias are called filters and represent particular features of the input (e.g., a particular shape). A distinguishing feature of CNNs is that many neurons can share the same filter. This reduces memory footprint because a single bias and a single vector of weights is used across all receptive fields sharing that filter, as opposed to each receptive field having its own bias and vector weighting

#### Architecture of CNN
![CNN Arch](https://user-images.githubusercontent.com/42957722/59995160-d32d2c00-966f-11e9-8d72-e95b750bc98a.png)
