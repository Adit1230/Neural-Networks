import math
import numpy as np
import csv
import matplotlib.pyplot as plt

class Neural_network():

    def sigmoid(x):
        return( (np.tanh(x/2) + 1) / 2 )
    
    def sigmoid_prime_inv(y):
        ans = np.zeros((len(y), len(y)))
        for i in range(len(y)):
            ans[i, i] = y[i] - y[i]**2
        return(ans)
    
    def ReLU(x):
        return( np.maximum(0, x) )
    
    def ReLU_prime_inv(y):
        ans = np.zeros((len(y), len(y)))
        for i in range(len(y)):
            ans[i, i] = int(y[i] > 0)
        return(ans)
        
    
    def softmax(x):
        x = x - ((max(x)//700) * 700)
        y = np.exp(x) / sum( np.exp(x) )
        return(y)
    
    def softmax_prime_inv(y):
        ans = np.outer(y, y) * -1
        for i in range(len(y)):
            ans[i, i] = y[i] - y[i] ** 2
        return(ans)

    def __init__(self, layers, outputs, logsize, learning_rate = 1, activations = 'sigmoid', activ_deriv_inv = 'sigmoid'):
        #Creates a neural network with layer sizes mentioned in layers
        #For example if layer is [4,3,2] the neural network created has 4 neurons in input layer, 3 neurons in a hidden layer and 2 neurons in the output layer
        #Activations is a list of the activation functions for each layer
        #Providing a single function name will set activation functions of all layers to that function
        #If f(x) is the activation function, activ_deriv_inv is f'( f inverse ( x ) )
        #Outputs is the list of outputs to be displayed if the corresponding output neron is activated
        
        self.example_no = 0
        self.logsize = logsize
        self.batch_cost_sum = 0
        self.batch_corr_predict = 0
        self.cost_log = {}
        self.accuracy_log = {}
        self.outputs = outputs
        self.learning_rate = learning_rate

        self.activations = []
        self.activation_names = []
        self.activ_deriv_inv = []

        if type(activations) == str:
            if activations.lower() == 'sigmoid':
                for layer in range(len(layers) - 1):
                    self.activation_names.append(activations)
                    self.activations.append( Neural_network.sigmoid )
                    self.activ_deriv_inv.append( Neural_network.sigmoid_prime_inv )
            
            elif activations.lower() == 'relu':
                for layer in range(len(layers) - 1):
                    self.activation_names.append(activations)
                    self.activations.append( Neural_network.ReLU )
                    self.activ_deriv_inv.append( Neural_network.ReLU_prime_inv )
                
            elif activations.lower() == 'softmax':
                for layer in range(len(layers) - 1):
                    self.activation_names.append(activations)
                    self.activations.append( Neural_network.softmax )
                    self.activ_deriv_inv.append( Neural_network.softmax_prime_inv )

        else:
            for function in activations:
                if function.lower() == 'sigmoid':
                    self.activations.append( Neural_network.sigmoid )
                    self.activ_deriv_inv.append( Neural_network.sigmoid_prime_inv )
                
                elif function.lower() == 'relu':
                    self.activations.append( Neural_network.ReLU )
                    self.activ_deriv_inv.append( Neural_network.ReLU_prime_inv )

                elif function.lower() in ('softmax', 'softargmax'):
                    self.activations.append( Neural_network.softmax )
                    self.activ_deriv_inv.append( Neural_network.softmax_prime_inv )

            self.activation_names = activations
        
        self.biases = []
        self.weights = []
        
        for layer in range(1, len(layers)):
            self.biases.append( np.zeros(layers[layer]) )
            self.weights.append( np.random.randn(layers[layer], layers[layer-1]) )
            
    def backpropogate(self, inputs, answer):
        layers = [np.array(inputs) / max(inputs),]

        for layer in range(len(self.biases)):
            layers.append(self.activations[layer]( (self.weights[layer] @ layers[layer]) + self.biases[layer]  ))

        if answer not in self.outputs:
            answer = int(answer)
            if answer not in self.outputs:
                answer = float(answer)
        expectation = np.zeros(len(self.outputs))
        expectation [self.outputs.index(answer)] = 1

        self.example_no += 1
        cost = sum((layers[-1] - expectation) ** 2) / len(self.outputs)
        self.batch_cost_sum += cost
        
        if self.outputs [np.where(layers[-1] == max(layers[-1]))[0][0]] == answer:
            self.batch_corr_predict += 1

        if self.example_no % self.logsize == 0:
            if self.cost_log == {}:
                batch_size = self.example_no
            else:
                batch_size = self.example_no - list(self.cost_log.keys())[-1]
            
            self.cost_log [self.example_no] = self.batch_cost_sum / batch_size
            self.batch_cost_sum = 0

            self.accuracy_log [self.example_no] = 100 * self.batch_corr_predict / batch_size
            self.batch_corr_predict = 0
        
        n = len(self.biases)
        bias_gradient = [ (self.activ_deriv_inv[-1](layers[n])) @ (2 * (layers[n] - expectation))]
        weights_gradient = [ np.outer(bias_gradient[0], layers[n-1]), ]
        
        for layer in range(1, n):
            bias_gradient.append( (self.activ_deriv_inv[n - layer - 1]( layers[n - layer] )) @ (bias_gradient[layer - 1] @ self.weights[n - layer]) )
            weights_gradient.append( np.outer( bias_gradient[layer], layers[n - layer - 1]) )

        bias_gradient.reverse()
        weights_gradient.reverse()

        return((bias_gradient, weights_gradient))

    def train(self, data, label = 'first'):
        n = len(self.biases)
        batch_size = len(data)
        bias_gradient = []
        weights_gradient = []
        
        for layer in range(n):
            bias_gradient.append( np.zeros(self.biases[layer].shape) )
            weights_gradient.append( np.zeros(self.weights[layer].shape) )

        if label == 'first':
            label = 0
            data_start = 1
            data_end = len(data[0])
        elif label == 'last':
            label = len(data[0]) - 1
            data_start = 0
            data_end = label
            
        for example in range(len(data)):
            bias_sensitivity, weights_sensitivity = self.backpropogate( data[example][data_start : data_end], data[example][label] )
            
            for layer in range(n):
                bias_gradient[layer] += bias_sensitivity[layer]
                weights_gradient[layer] += weights_sensitivity[layer]

        for layer in range(n):
                self.biases[layer] += (bias_gradient[layer] / batch_size) * - self.learning_rate
                self.weights[layer] += (weights_gradient[layer] / batch_size) * - self.learning_rate

        return()

    def train_from_file(self, file, batch_size, label = 'first', start = 'start', end = 'end', seperator = ',', headers = True):
        n = len(self.biases)
        
        bias_gradient = []
        weights_gradient = []
        for layer in range(n):
                bias_gradient.append( np.zeros(self.biases[layer].shape) )
                weights_gradient.append( np.zeros(self.weights[layer].shape) )
        
        convert_to_float = np.vectorize( lambda x: float(x) )
        
        with open(file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = seperator)

            if start == 'start':
                start = 0
            if end == 'end':
                end = len(csv_file.readlines())
                csv_file.seek(0)

                if headers == True:
                    end -= 1

            next(csv_reader)
            fields_no = len(next(csv_reader))
            csv_file.seek(0)

            if label == 'first':
                label = 0
                data_start = 1
                data_end = fields_no
            elif label == 'last':
                label = fields_no - 1
                data_start = 0
                data_end = label

            for i in range(start):
                next(csv_reader)
            
            if headers == True:
                next(csv_reader)

            for example_no in range(start, end):
                example = next(csv_reader)

                if example[label] not in self.outputs:
                    example[label] = eval(example[label])
                    if example[label] not in self.outputs:
                        example[label] = eval(example[label])

                bias_sensitivity, weights_sensitivity = self.backpropogate( convert_to_float(example[data_start : data_end]), example[label] )
                
                for layer in range(n):
                    bias_gradient[layer] += bias_sensitivity[layer]
                    weights_gradient[layer] += weights_sensitivity[layer]

                if (example_no - start) % batch_size == 0:
                    for layer in range(n):
                        self.biases[layer] += (bias_gradient[layer] / batch_size) * - self.learning_rate
                        self.weights[layer] += (weights_gradient[layer] / batch_size) * - self.learning_rate

                    bias_gradient = []
                    weights_gradient = []
                    for layer in range(n):
                        bias_gradient.append( np.zeros(self.biases[layer].shape) )
                        weights_gradient.append( np.zeros(self.weights[layer].shape) )

        return()

    def test_from_file(self, file, label = 'first', start = 'start', end = 'end', seperator = ',', headers = True):
        n = len(self.biases)
        correct_predictions = 0

        convert_to_float = np.vectorize( lambda x: float(x) )
        
        with open(file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = seperator)

            if start == 'start':
                start = 0
            if end == 'end':
                end = len(csv_file.readlines())
                csv_file.seek(0)

                if headers == True:
                    end -= 1

            next(csv_reader)
            fields_no = len(next(csv_reader))
            csv_file.seek(0)

            if label == 'first':
                label = 0
                data_start = 1
                data_end = fields_no
            elif label == 'last':
                label = fields_no - 1
                data_start = 0
                data_end = label

            for i in range(start):
                next(csv_reader)
            
            if headers == True:
                next(csv_reader)

            for example_no in range(start, end):
                example = next(csv_reader)

                answer = example[label]
                
                if answer not in self.outputs:
                    answer = eval(example[label])
                    if answer not in self.outputs:
                        answer = eval(example[label])

                if answer not in self.outputs:
                    answer = int(answer)
                    if answer not in self.outputs:
                        answer = float(answer)

                prediction, confidence = self.predict( convert_to_float( example[data_start : data_end] ) )

                if prediction == answer:
                    correct_predictions += 1

            return( 100 * correct_predictions / (end - start) )

    def predict(self, inputs):
        neurons = np.array(inputs) / max(inputs)
        
        for layer in range(len(self.biases)):
            neurons = self.activations[layer]( (self.weights[layer] @ neurons) + self.biases[layer]  )

        maxval = max(neurons)
        totalval = sum(neurons)
        result = np.where(neurons == maxval)[0][0]

        return( (self.outputs[result], 100 * maxval/totalval) )

    def save(self, filename):
        with open(filename, 'w', newline = '') as file:
            csv_writer = csv.writer(file, delimiter = '\t', quoting = csv.QUOTE_NONNUMERIC)

            csv_writer.writerow( [self.example_no, self.logsize, self.batch_cost_sum, self.batch_corr_predict, str(self.cost_log), str(self.accuracy_log)] )
            csv_writer.writerow( [self.learning_rate, ] )
            csv_writer.writerow( self.activation_names )
            csv_writer.writerow( self.outputs )

            biases = []
            for bias in self.biases:
                biases.append(str(list(bias)))
            weights = []
            for weight in self.weights:
                weights.append(str(weight.tolist()))
            
            csv_writer.writerow( biases )
            csv_writer.writerow( weights )

    @classmethod
    def load(cls, filename):
        field_size = 131072
        while True:
            try:
                with open(filename, 'r') as file:
                    csv_reader = csv.reader(file, delimiter = '\t', quoting = csv.QUOTE_NONNUMERIC)
                    model_data = []
                    for line in csv_reader:
                        model_data.append(line)
                break
            
            except csv.Error:
                field_size = 2 * field_size
                csv.field_size_limit(field_size)

        csv.field_size_limit(131072)
            
        model = cls([2, 2], [0, 1], 10, activations = model_data[2])
        
        model.example_no = int(model_data[0][0])
        model.logsize = int(model_data[0][1])
        model.batch_cost_sum = float(model_data[0][2])
        model.batch_corr_predict = int(model_data[0][3])
        model.cost_log = eval(model_data[0][4])
        model.accuracy_log = eval(model_data[0][5])
        model.learning_rate = float(model_data[1][0])
        model.outputs = model_data[3]

        model.biases = []
        model.weights = []

        for bias in model_data[4]:
            model.biases.append( np.array(eval(bias)) )

        for weight in model_data[5]:
            model.weights.append( np.array(eval(weight)) )

        return(model)

    def plot_stats(self, mean_sq_error, accuracy):
        title = ''
        ylabel = ''
        
        if mean_sq_error == True:
            plt.plot( list(self.cost_log.keys()), list(self.cost_log.values()), color = 'r', marker = 'o', label = "Mean squared error" )
            title = "Mean squared error of neural network"
            ylabel = "Mean squared error"

        if accuracy == True:
            plt.plot( list(self.accuracy_log.keys()), list(self.accuracy_log.values()), color = 'g', marker = 'o', label = "Accuracy" )
            title = "Accuracy of neural network"
            ylabel = "Accuracy in percentage"

        if mean_sq_error == True and accuracy == True:
            title = "Mean squared error and accuracy of neural network"
            ylabel = "Mean squared error / Accuracy in percentage"

        plt.title(title)
        plt.xlabel("No. of training examples")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        plt.show()
        
