from random import uniform


and_examples = [ [0,0,0],
                 [0,1,0],
                 [1,0,0],
                 [1,1,1] ]
                 
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
        self._alpha = 0.4 #learning rate
    
    def normalize(self, output):
        if(output > 0):
            return 1
        return 0
        
    def calculate_output(self, example):
        return self._w1*get_x1(example) + self._w2*get_x2(example) + self._b

    def update(self, example):
        output = self.calculate_output(example)
        if(self.normalize(output) != get_y(example)):
            self.update_weights(example, output)
        
    def update_weights(self, example, output):
        delta = get_y(example) - output
        self._w1 += self._alpha * delta * get_x1(example)
        self._w2 += self._alpha * delta * get_x2(example)
        self._b += self._alpha * delta
        

def init_perceptron():
    w1 = uniform(-0.99, 0.99)
    w2 = uniform(-0.99, 0.99)
    b = uniform(-0.99, 0.99)
    # w1 = 0.5
    # w2 = 0.5
    # b = 0.5
    return Perceptron(w1, w2, b)
        
def learn(perceptron):
    for i in range(10):
        for example in and_examples:
            perceptron.update(example)
        
def test(perceptron):
    for example in and_examples:
        print_example(example)
        print(" got: "+str( perceptron.normalize(perceptron.calculate_output(example)) ))
    
def print_example(example):
    print(str(get_x1(example)) + " AND "+str(get_x2(example))+" = "+str(get_y(example)) )
    
    
if __name__ == "__main__":
    perceptron = init_perceptron()
    learn(perceptron)
    test(perceptron)
