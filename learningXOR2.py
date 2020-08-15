from random import uniform
from math import e
import matplotlib.pyplot as plt

xor_examples = [ [0,0,0],
                 [0,1,1],
                 [1,0,1],
                 [1,1,0] ]
to_plot = []
alpha = 0.5 #learning rate
iterations = 10000

def get_x1(example):
    return example[0]
def get_x2(example):
    return example[1]
def get_y(example):
    return example[2]
    
def activation(x):
    return sigmoid(x)
def d_activation(x):
    return d_sigmoid(x)
def sigmoid(x):
    return 1 / (1 + e**(-x))
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Perceptron:
    def __init__(self, w1, w2, b):
        # print(w1, w2, b)
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.adj_p1 = None
        self.adj_p2 = None
        self.reset_weight_deltas()
        
    def reset_weight_deltas(self):
        self.dw1 = 0
        self.dw2 = 0
        self.db = 0
        
    def add_adjacent_perceptrons(self, p1, p2):
        self.adj_p1 = p1
        self.adj_p2 = p2
        
    def output(self, example):
        if(self.adj_p1 != None):
            x1 = activation(self.adj_p1.output(example))
            x2 = activation(self.adj_p2.output(example))
        else:
            x1 = get_x1(example)
            x2 = get_x2(example)
        return self.w1*x1 + self.w2*x2 + self.b 

    def learn(self, example):
        output = self.output(example)
        z      = activation(output) 
        y      = get_y(example)
        delta  = z - y
        chain  = alpha * delta
        self.update_weights_deltas(example, chain)
        return self.error(z, y)
        
    def update_weights_deltas(self, example, chain):
        chain *= d_activation(self.output(example)) 
        if(self.adj_p1 != None):
            self.dw1 += chain * activation(self.adj_p1.output(example))
            self.dw2 += chain * activation(self.adj_p2.output(example))
            self.adj_p1.update_weights_deltas(example, chain * self.w1)
            self.adj_p2.update_weights_deltas(example, chain * self.w2)
        else:
            self.dw1 += chain * get_x1(example)
            self.dw2 += chain * get_x2(example)
        self.db += chain
        
    def update(self):
        # print(self.dw1/len(xor_examples))
        # print(self.dw2/len(xor_examples))
        # print(self.db/len(xor_examples))
        self.w1 -= self.dw1/len(xor_examples)
        self.w2 -= self.dw2/len(xor_examples)
        self.b  -= self.db/len(xor_examples)
        if(self.adj_p1 != None):
            self.adj_p1.update()
            self.adj_p2.update()
        self.reset_weight_deltas()
        
    def error(self, z, y):
        return self.se(z, y)
    def percent_error(self, z, y):
        return abs(z-y)*100
    def se(self, z, y):
        return 0.5*(z-y)**2
        
    def print(self, tab = 0):
        start = " " * tab
        print(start + "w1: "+ str(self.w1))
        print(start + "w2: "+ str(self.w2))
        print(start + "b: "+str(self.b))
        if(self.adj_p1 != None):
            self.adj_p1.print(2)
            self.adj_p2.print(2)
    
def init_MLP():
    a = Perceptron(uniform(-1,1)/3, uniform(-1,1)/3, uniform(-1,1)/3)
    b = Perceptron(uniform(-1,1)/3, uniform(-1,1)/3, uniform(-1,1)/3)
    out = Perceptron(uniform(-1,1)/3, uniform(-1,1)/3, uniform(-1,1)/3)
    out.add_adjacent_perceptrons(a,b)
    return out
        
def learn(mlp):
    for i in range(iterations):
        avg_error = 0
        for example in xor_examples:
            avg_error += mlp.learn(example)
        avg_error /= len(xor_examples)
        to_plot.append(avg_error)
        mlp.update()
        
def test(mlp):
    avg_error = 0
    for example in xor_examples:
        print_example(example)
        output = mlp.output(example)
        z      = activation(output)
        y      = get_y(example)
        avg_error += mlp.error(z, y)
        print(" ["+str(z) +"]")
    avg_error /= len(xor_examples)
    to_plot.append(avg_error)
    
def print_example(example):
    print(str(get_x1(example)) + " XOR "+str(get_x2(example))+" = "+str(get_y(example)), end="")
    
def plot(current):
    plt.close()
    plt.plot([i for i in range(len(to_plot))], to_plot, '-o')
    # plt.ylim(0,100)
    plt.savefig("error_rate"+str(current)+".png")
    
def run(current):
    global to_plot
    print(" "+str(current)+" =====================================================")
    to_plot = []
    mlp = init_MLP()
    learn(mlp)
    test(mlp)
    plot(current)
    
if __name__ == "__main__":
    for i in range(10):
        run(i)
    # run(0)
