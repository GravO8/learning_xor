from random import uniform
from math import e
import matplotlib.pyplot as plt

xor_examples = [ [0,0,0],
                 [0,1,1],
                 [1,0,1],
                 [1,1,0] ]
to_plot = []
current = 0
alpha = 0.01 #learning rate
iterations = 10000
                 
def get_x1(example):
    return example[0]
def get_x2(example):
    return example[1]
def get_y(example):
    return example[2]


class Perceptron:
    def __init__(self, w1, w2, b):
        self._w1 = w1
        self._w2 = w2
        self._b = b
        self._adj_p1 = None
        self._adj_p2 = None
        
    def add_adjacent_perceptrons(self, p1, p2):
        self._adj_p1 = p1
        self._adj_p2 = p2
    
    def activation(self, x):
        return self.sigmoid(x)
    def d_activation(self, x):
        return self.d_sigmoid(x)
    def sigmoid(self, x):
        return e**x / (e**x + 1)
    def d_sigmoid(self, x):
        return e**x / ((e**x + 1)**2)
        
    def output(self, example):
        x1 = 0
        x2 = 0
        if(self._adj_p1 != None):
            x1 = self._adj_p1.output(example)
            x2 = self._adj_p2.output(example)
        else:
            x1 = get_x1(example)
            x2 = get_x2(example)
        return self._w1*x1 + self._w2*x2 + self._b

    def update(self, example):
        output = self.output(example)
        y      = get_y(example)
        delta  = y - self.activation(output)
        chain  = alpha * delta 
        self.update_weights(example, chain)
        return self.error(self.activation(output), y)
        
    def update_weights(self, example, chain):
        chain *= self.d_activation(self.output(example)) 
        if(self._adj_p1 != None):
            self._w1 += chain * self.activation(self._adj_p1.output(example))
            self._w2 += chain * self.activation(self._adj_p2.output(example))
            self._adj_p1.update_weights(example, chain * self._w1)
            self._adj_p2.update_weights(example, chain * self._w2)
        else:
            self._w1 += chain * get_x1(example)
            self._w2 += chain * get_x2(example)
        self._b += chain
        
    def error(self, output, y):
        return self.se(output, y)
    def percent_error(self, output, y):
        return abs(self.activation(output)-y)*100
    def se(self, output, y):
        return (self.activation(output)-y)**2
        

def init_perceptron():
    w1 = uniform(-0.99, 0.99)/3
    w2 = uniform(-0.99, 0.99)/3
    b = uniform(-0.99, 0.99)/3
    return Perceptron(w1, w2, b)
    
def init_MLP():
    a = init_perceptron()
    b = init_perceptron()
    out = init_perceptron()
    out.add_adjacent_perceptrons(a,b)
    return out
        
def learn(mlp):
    for i in range(iterations):
        avg_error = 0
        for example in xor_examples:
            avg_error += mlp.update(example)
        avg_error /= len(xor_examples)
        to_plot.append(avg_error)
        
def test(mlp):
    avg_error = 0
    for example in xor_examples:
        print_example(example)
        output = mlp.output(example)
        z      = mlp.activation(output)
        y      = get_y(example)
        avg_error += mlp.error(z, y)
        print(" ["+str(z) +"]")
    avg_error /= len(xor_examples)
    to_plot.append(avg_error)
    
def print_example(example):
    print(str(get_x1(example)) + " XOR "+str(get_x2(example))+" = "+str(get_y(example)), end="")
    
def plot():
    plt.close()
    plt.plot([i for i in range(len(to_plot))], to_plot, '-o')
    # plt.ylim(0,100)
    plt.savefig("error_rate"+str(current)+".png")
    
def run():
    global current, to_plot
    print(" "+str(current)+" =====================================================")
    to_plot = []
    mlp = init_MLP()
    learn(mlp)
    test(mlp)
    plot()
    current += 1
    
if __name__ == "__main__":
    for i in range(10):
        run()
