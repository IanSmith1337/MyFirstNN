import numpy
# Neural Network Definitions

class neural:
    numpy.random.seed(12345)
    
    def __init__(self):
        self.inputs = 2
        self.outputs = 2
        self.hidden = 3
        self.batch_size = 1
        self.raw = None
        self.hidden_vals = None
        self.weight1 = None
        self.weight2 = None
        self.hr = None
        self.output_vals = None

    def printRaw(self):
        self.raw = numpy.random.randn(self.batch_size, self.inputs)
        print(self.raw)
        
    def printWeights(self):
        self.weight1 = numpy.random.randn(self.inputs, self.hidden)
        self.weight2 = numpy.random.randn(self.hidden, self.outputs)
        print(self.weight1)
        print(self.weight2)
        
    def RELU(self):
        self.hidden_vals = self.raw.dot(self.weight1)
        self.hr = numpy.maximum(self.hidden_vals, 0)
        print(self.hr)
        
    def printOutput(self):
        self.output_vals = self.hr.dot(self.weight2)
        numpy.set_printoptions(precision=2)
        print(self.output_vals)
            