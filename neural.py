import numpy
# Neural Network Definitions

# Variable defs.
inputs = 2
outputs = 2
hidden = 3
batch_size = 10
raw = numpy.random.randn(batch_size, inputs)

def printRaw():
    print(raw)