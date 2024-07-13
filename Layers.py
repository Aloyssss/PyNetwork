import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        """
        Compute the layer output for a given input
        """
        raise NotImplementedError

    def back_propagation(self, output_error, learning_rate):
        """
        Compute the next layer error (backpropagation) and updates the parameters
        """
        raise NotImplementedError


class Dense(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def back_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    def back_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
    
    
class RNN(Layer):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weights_input = np.random.rand(input_size, hidden_size) - 0.5
        self.weights_hidden = np.random.rand(hidden_size, hidden_size) - 0.5
        self.weights_output = np.random.rand(hidden_size, output_size) - 0.5
        self.bias_hidden = np.random.rand(1, hidden_size) - 0.5
        self.bias_output = np.random.rand(1, output_size) - 0.5
        self.hidden_state = np.zeros((1, hidden_size))

    def forward_propagation(self, input_data):
        self.input = input_data
        self.hidden_state = np.tanh(np.dot(input_data, self.weights_input) + np.dot(self.hidden_state, self.weights_hidden) + self.bias_hidden)
        self.output = np.dot(self.hidden_state, self.weights_output) + self.bias_output
        return self.output

    def back_propagation(self, output_error, learning_rate):
        output_error = np.atleast_2d(output_error)
        hidden_error = np.dot(output_error, self.weights_output.T) * (1 - self.hidden_state ** 2)
        
        weights_output_error = np.dot(self.hidden_state.T, output_error)
        weights_hidden_error = np.dot(self.hidden_state.T, hidden_error)
        weights_input_error = np.dot(self.input.T, hidden_error)
        
        self.weights_output -= learning_rate * weights_output_error
        self.weights_hidden -= learning_rate * weights_hidden_error
        self.weights_input -= learning_rate * weights_input_error
        self.bias_output -= learning_rate * output_error
        self.bias_hidden -= learning_rate * hidden_error

        return np.dot(hidden_error, self.weights_input.T)