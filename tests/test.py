inputs = [1, 2, 3, 2.5]

# Every input will have a unique weigh associated to it
weights = [0.2, 0.8, -0.5, 1.0]

# Every unique neuron has a unique bias
bias = 2


output = 0
for i in range(len(inputs)):
    output += inputs[i] * weights[i]

output += bias

print(output)