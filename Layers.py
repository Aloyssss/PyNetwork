import numpy as np
from Activation import sigmoid, sigmoid_prime, tanh, tanh_prime
from Weights import xavier_norm

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

    def forward_propagation(self, input_data, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.hidden_state
        self.input = input_data
        self.hidden_state = np.tanh(np.dot(input_data, self.weights_input) + np.dot(hidden_state, self.weights_hidden) + self.bias_hidden)
        self.output = np.dot(self.hidden_state, self.weights_output) + self.bias_output
        return self.output

    def back_propagation(self, output_error, learning_rate, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.hidden_state
        output_error = np.atleast_2d(output_error)
        hidden_error = np.dot(output_error, self.weights_output.T) * (1 - self.hidden_state ** 2)
        
        weights_output_error = np.dot(self.hidden_state.T, output_error)
        weights_hidden_error = np.dot(hidden_state.T, hidden_error)
        weights_input_error = np.dot(self.input.T, hidden_error)
        
        self.weights_output -= learning_rate * weights_output_error
        self.weights_hidden -= learning_rate * weights_hidden_error
        self.weights_input -= learning_rate * weights_input_error
        self.bias_output -= learning_rate * output_error
        self.bias_hidden -= learning_rate * hidden_error

        return np.dot(hidden_error, self.weights_input.T)


class LSTM(Layer):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Weights and biases initialization
        self.Wf = xavier_norm(input_size, hidden_size)  # forget gate weights
        self.Wi = xavier_norm(input_size, hidden_size)  # input gate weights
        self.Wo = xavier_norm(input_size, hidden_size)  # output gate weights
        self.Wc = xavier_norm(input_size, hidden_size)  # candidate gate weights

        self.bf = np.zeros((1, hidden_size))  # forget gate bias
        self.bi = np.zeros((1, hidden_size))  # input gate bias
        self.bo = np.zeros((1, hidden_size))  # output gate bias
        self.bc = np.zeros((1, hidden_size))  # cell bias

        self.hidden_state = np.zeros((hidden_size, 1))
        self.cell_state = np.zeros((hidden_size, 1))

    def reset(self):
        self.concat_self.input = {}

        self.hidden_states = {-1:np.zeros((self.hidden_size, 1))}
        self.cell_states = {-1:np.zeros((self.hidden_size, 1))}

        self.activation_outputs = {}
        self.candidate_gates = {}
        self.output_gates = {}
        self.forget_gates = {}
        self.input_gates = {}
        self.outputs = {}

    def forward_propagation(self, input_data):
        
        self.reset()

        self.input = input_data

        for q in range(len(self.input)):
            self.concat_input[q] = np.concatenate((self.hidden_states[q - 1], self.input[q]))

            self.forget_gates[q] = sigmoid(np.dot(self.wf, self.concat_inputs[q]) + self.bf)
            self.input_gates[q] = sigmoid(np.dot(self.wi, self.concat_inputs[q]) + self.bi)
            self.candidate_gates[q] = tanh(np.dot(self.wc, self.concat_inputs[q]) + self.bc)
            self.output_gates[q] = sigmoid(np.dot(self.wo, self.concat_inputs[q]) + self.bo)

            self.cell_states[q] = self.forget_gates[q] * self.cell_states[q - 1] + self.input_gates[q] * self.candidate_gates[q]
            self.hidden_states[q] = self.output_gates[q] * tanh(self.cell_states[q])

            outputs += [np.dot(self.wy, self.hidden_states[q]) + self.by]
        return outputs

    def back_propagation(self, output_error, learning_rate):
        d_wf, d_bf = 0, 0
        d_wi, d_bi = 0, 0
        d_wc, d_bc = 0, 0
        d_wo, d_bo = 0, 0
        d_wy, d_by = 0, 0

        dh_next, dc_next = np.zeros_like(self.hidden_states[0]), np.zeros_like(self.cell_states[0])
        for q in reversed(range(len(self.input))):
            error = output_error[q]

            # Final Gate Weights and Biases Errors
            d_wy += np.dot(error, self.hidden_states[q].T)
            d_by += error

            # Hidden State Error
            d_hs = np.dot(self.wy.T, error) + dh_next

            # Output Gate Weights and Biases Errors
            d_o = tanh(self.cell_states[q]) * d_hs * sigmoid(self.output_gates[q], derivative = True)
            d_wo += np.dot(d_o, self.input[q].T)
            d_bo += d_o

            # Cell State Error
            d_cs = tanh(tanh(self.cell_states[q]), derivative = True) * self.output_gates[q] * d_hs + dc_next

            # Forget Gate Weights and Biases Errors
            d_f = d_cs * self.cell_states[q - 1] * sigmoid(self.forget_gates[q], derivative = True)
            d_wf += np.dot(d_f, self.input[q].T)
            d_bf += d_f

            # Input Gate Weights and Biases Errors
            d_i = d_cs * self.candidate_gates[q] * sigmoid(self.input_gates[q], derivative = True)
            d_wi += np.dot(d_i, self.input[q].T)
            d_bi += d_i
            
            # Candidate Gate Weights and Biases Errors
            d_c = d_cs * self.input_gates[q] * tanh(self.candidate_gates[q], derivative = True)
            d_wc += np.dot(d_c, self.input[q].T)
            d_bc += d_c

            # Concatenated Input Error (Sum of Error at Each Gate!)
            d_z = np.dot(self.wf.T, d_f) + np.dot(self.wi.T, d_i) + np.dot(self.wc.T, d_c) + np.dot(self.wo.T, d_o)

            # Error of Hidden State and Cell State at Next Time Step
            dh_next = d_z[:self.hidden_size, :]
            dc_next = self.forget_gates[q] * d_cs

        for d_ in (d_wf, d_bf, d_wi, d_bi, d_wc, d_bc, d_wo, d_bo, d_wy, d_by):
            np.clip(d_, -1, 1, out = d_)

        self.wf += d_wf * self.learning_rate
        self.bf += d_bf * self.learning_rate

        self.wi += d_wi * self.learning_rate
        self.bi += d_bi * self.learning_rate

        self.wc += d_wc * self.learning_rate
        self.bc += d_bc * self.learning_rate

        self.wo += d_wo * self.learning_rate
        self.bo += d_bo * self.learning_rate

        self.wy += d_wy * self.learning_rate
        self.by += d_by * self.learning_rate