from tqdm import tqdm
from Layers import RNN, LSTM

class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.loss_history = []

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        raise NotImplemented

    def fit(self, x_train, y_train, epochs, learning_rate):
        raise NotImplemented

class FeedForward(Network):
    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in tqdm(range(epochs), desc="Epochs", total=epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.back_propagation(error, learning_rate)

            err /= samples
            self.loss_history.append(err)
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

class Sequential(Network):
    def __init__(self):
        super().__init__()
        self.hidden_state = None
        self.cell_state = None

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                if isinstance(layer, RNN):
                    output = layer.forward_propagation(output, self.hidden_state)
                    self.hidden_state = layer.hidden_state
                if isinstance(layer, LSTM):
                    output = layer.forward_propagation(output, self.hidden_state, self.cell_state)
                    self.hidden_state = layer.hidden_state
                    self.cell_state = layer.cell_state
                else:
                    output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in tqdm(range(epochs), desc="Epochs", total=epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    if isinstance(layer, RNN):
                        output = layer.forward_propagation(output, self.hidden_state)
                        self.hidden_state = layer.hidden_state
                    if isinstance(layer, LSTM):
                        raise NotImplemented
                    else:
                        output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    if isinstance(layer, RNN):
                        error = layer.back_propagation(error, learning_rate, self.hidden_state)
                    if isinstance(layer, LSTM):  # Handle LSTM layer
                        output = layer.forward_propagation(output, self.hidden_state, self.cell_state)
                        self.hidden_state = layer.hidden_state
                        self.cell_state = layer.cell_state
                    else:
                        error = layer.back_propagation(error, learning_rate)

            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

