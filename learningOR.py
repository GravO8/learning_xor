from random import uniform
import matplotlib.pyplot as plt

or_examples = [ [0,0,0],
                [0,1,1],
                [1,0,1],
                [1,1,1] ]
to_plot = []
current = 0
iterations = 20
                 
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
        self._alpha = 0.15 #learning rate
    
    def normalize(self, output):
        if(output > 0):
            return 1
        return 0
        
    def calculate_output(self, example):
        return self._w1*get_x1(example) + self._w2*get_x2(example) + self._b

    def update(self, example):
        output = self.normalize(self.calculate_output(example))
        y = get_y(example)
        if(output != y):
            self.update_weights(example, output)    
        return self.percent_error(output, y)
        
    def update_weights(self, example, output):
        delta = get_y(example) - output
        self._w1 += self._alpha * delta * get_x1(example)
        self._w2 += self._alpha * delta * get_x2(example)
        self._b += self._alpha * delta
        
    def percent_error(self, output, y):
        return abs(self.normalize(output)-y)*100
        

def init_perceptron():
    w1 = uniform(-0.99, 0.99)
    w2 = uniform(-0.99, 0.99)
    b = uniform(-0.99, 0.99)
    # w1 = 0.5
    # w2 = 0.5
    # b = 0.5
    return Perceptron(w1, w2, b)
        
def learn(perceptron):
    for i in range(iterations):
        avg_error = 0
        for example in or_examples:
            avg_error += perceptron.update(example)
        avg_error /= len(or_examples)
        to_plot.append(avg_error)
        
def test(perceptron):
    avg_error = 0
    for example in or_examples:
        print_example(example)
        y = get_y(example)
        output = perceptron.calculate_output(example)
        avg_error += perceptron.percent_error(output, y)
        print(" ["+str( perceptron.normalize(output) ) +"]")
    avg_error /= len(or_examples)
    to_plot.append(avg_error)
    
def print_example(example):
    print(str(get_x1(example)) + " OR "+str(get_x2(example))+" = "+str(get_y(example)), end="")
    
def plot():
    plt.close()
    plt.plot([i for i in range(len(to_plot))], to_plot, '-o')
    plt.ylim(0,100)
    # plt.show()
    plt.savefig("error_rate"+str(current)+".png")
    
def run():
    global current, to_plot
    print(" "+str(current)+" =====================================================")
    to_plot = []
    perceptron = init_perceptron()
    learn(perceptron)
    test(perceptron)
    plot()
    current += 1
    
if __name__ == "__main__":
    for i in range(10):
        run()
