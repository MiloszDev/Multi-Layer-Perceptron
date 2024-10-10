class Neuron:
    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
    
    def forward(self):
        self.output = 0

        for i in range(len(self.inputs)):
            self.output += self.inputs[i] * self.weights[i]

        return self.output + self.bias    



neuron1 = Neuron(inputs=[1, 2, 3, 2.5], weights=[0.2, 0.8, -0.5, 1.0], bias=2).forward()
neuron2 = Neuron(inputs=[1, 2, 3, 2.5], weights=[0.5, -0.91, 0.26, -0.5], bias=3).forward()
neuron3 = Neuron(inputs=[1, 2, 3, 2.5], weights=[-0.26, -0.27, 0.17, 0.87], bias=0.5).forward()

print([neuron1, neuron2, neuron3])