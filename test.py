from learningXOR2 import Perceptron
from learningXOR2 import sigmoid
from learningXOR2 import d_sigmoid
from learningXOR2 import activation
from learningXOR2 import d_activation
from learningXOR2 import alpha


xor_examples = [ [0,0,0],
                 [0,1,1],
                 [1,0,1],
                 [1,1,0] ]
                 
def get_x1(example):
    return example[0]
def get_x2(example):
    return example[1]
def get_y(example):
    return example[2]

def run_tests():
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    print("all tests gud")
    
epsilon = 0.000001

def test1():
    print("running test1")
    p = Perceptron(1,1,1)
    output = p.output(xor_examples[0])
    if(output != 1):
        raise ValueError("test1 failed") 
    
def test2():
    print("running test2")
    p = Perceptron(1,1,1)
    z = activation(p.output(xor_examples[0]))
    if(abs(z - 0.7310585786300049) > epsilon):
        raise ValueError("test2 failed") 
        
def test3():
    print("running test3")
    a = Perceptron(1,1,1)
    b = Perceptron(1,1,1)
    p = Perceptron(1,1,1)
    p.add_adjacent_perceptrons(a,b)
    output = p.output(xor_examples[0])
    if(abs(output - 2.4621171572600096) > epsilon):
        raise ValueError("test3 failed") 
        
def test4():
    print("running test4")
    a = Perceptron(1,1,1)
    b = Perceptron(1,1,1)
    p = Perceptron(1,1,1)
    p.add_adjacent_perceptrons(a,b)
    z = activation(p.output(xor_examples[0]))
    if(abs(z - 0.9214430516601156) > epsilon):
        raise ValueError("test4 failed")
        
def test5():
    print("running test5")
    p       = Perceptron(1,1,1)
    example = xor_examples[2]
    s       = p.output(example)
    z       = activation(s)
    y       = get_y(example)
    updateW1 = alpha * (z - y) * d_activation(s) * get_x1(example) / len(xor_examples)
    p.learn(example)
    p.update()
    if(abs(p.w1 - (1 - updateW1)) > epsilon):
        raise ValueError("test5 failed. Expected: "+str(1 - updateW1)+" .Got: "+str(p.w1))
        
def test6():
    print("running test6")
    a = Perceptron(1,1,1)
    b = Perceptron(1,1,1)
    p = Perceptron(1,1,1)
    p.add_adjacent_perceptrons(a,b)
    example = xor_examples[2]
    s       = p.output(example)
    z       = activation(s)
    zA      = activation(p.adj_p1.output(example))
    y       = get_y(example)
    updateW1 = alpha * (z - y) * d_activation(s) * zA / len(xor_examples)
    p.learn(example)
    p.update()
    if(abs(p.w1 - (1 - updateW1)) > epsilon):
        raise ValueError("test6 failed. Expected: "+str(1 - updateW1)+" .Got: "+str(p.w1))

def test7():
    print("running test7")
    a = Perceptron(1,1,1)
    b = Perceptron(1,1,1)
    p = Perceptron(1,1,1)
    p.add_adjacent_perceptrons(a,b)
    example = xor_examples[2]
    s       = p.output(example)
    sA      = p.adj_p1.output(example)
    z       = activation(s)
    y       = get_y(example)
    updateWa1 = alpha * (z - y) * d_activation(s) * p.w1 * d_activation(sA) * get_x1(example) / len(xor_examples)
    p.learn(example)
    p.update()
    if(abs(p.adj_p1.w1 - (1 - updateWa1)) > epsilon):
        raise ValueError("test7 failed. Expected: "+str(1 - updateWa1)+" .Got: "+str(p.adj_p1.w1))
                
        
if __name__ == "__main__":
    run_tests()
