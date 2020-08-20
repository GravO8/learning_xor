from random import uniform
from math import e
import matplotlib.pyplot as plt

sub_examples = [ [0,1,-1],
                 [2,2,0],
                 [5,3,2],
                 [10,8,2],
                 [1,1,0],
                 [7,7,0],
                 [110,2,108],
                 [22,543,-521],
                 [0,0,0],
                 [37,23,14],
                 [20,23,-3] ]
                 
test_set = [ [-1,1,-2],
             [432,1,431],
             [-4,-4,0],
             [23,2,21],
             [3,-4,7],
             [333,222,111] ]
to_plot = []
alpha = 0.00007 #learning rate
decay = 0.05
iterations = 100

def get_x1(example):
    return example[0]
def get_x2(example):
    return example[1]
def get_y(example):
    return example[2]


class Perceptron:
    def __init__(self, w1, w2, b):
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.reset_weight_deltas()
        
    def reset_weight_deltas(self):
        self.dw1 = 0
        self.dw2 = 0
        self.db = 0
        
    def output(self, example):
        x1 = get_x1(example)
        x2 = get_x2(example)
        return self.w1*x1 + self.w2*x2 + self.b

    def learn(self, example):
        z      = self.output(example)
        y      = get_y(example)
        chain  = alpha * (z - y)
        # print(chain)
        self.update_weights_deltas(example, chain)
        return self.error(z, y)
        
    def update_weights_deltas(self, example, chain):
        self.dw1 += chain * get_x1(example)
        self.dw2 += chain * get_x2(example)
        self.db += chain
        
    def update(self):
        self.w1 = self.w1 - self.dw1/len(sub_examples)
        self.w2 = self.w2 - self.dw2/len(sub_examples)
        self.b  = self.b*(1 - decay) - self.db/len(sub_examples)
        self.reset_weight_deltas()
        
    def error(self, z, y):
        return self.se(z, y)
    def percent_error(self, z, y):
        return abs(z-y)*100
    def se(self, z, y):
        return 0.5*(z-y)**2
            
def init_MLP():
    return Perceptron(uniform(-1,1)/3, uniform(-1,1)/3, uniform(-1,1)/3)
        
def learn(mlp):
    for i in range(iterations):
        avg_error = 0
        for example in sub_examples:
            avg_error += mlp.learn(example)
        avg_error /= len(sub_examples)
        to_plot.append(avg_error)
        mlp.update()
        
def test(mlp):
    for example in test_set:
        print_example(example)
        print(" ["+str(round(mlp.output(example))) +"]")
    
def print_example(example):
    print(str(get_x1(example)) + " - "+str(get_x2(example))+" = "+str(get_y(example)), end="")
    
def plot(current):
    plt.close()
    plt.plot([i for i in range(len(to_plot))], to_plot, '-o')
    # plt.ylim(0,1)
    plt.savefig("error_rate"+str(current)+".png")
    
def run(current):
    global to_plot, alpha
    print(" "+str(current)+" alpha "+str(alpha)+" ======================================================")
    to_plot = []
    mlp = init_MLP()
    learn(mlp)
    print(str(mlp.w1)+"*x1 + "+str(mlp.w2)+"*x2 + "+str(mlp.b))
    test(mlp)
    plot(current)
    
if __name__ == "__main__":
    for i in range(10):
        run(i)
