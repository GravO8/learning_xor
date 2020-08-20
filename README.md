<h1 align="center">Learning simple binary functions</h1>

Here's a set of very simple programs that learn the AND, OR, ADD, SUB and XOR function. 

## Learning AND 

| x1 | x2 | y |
|:--:|:--:|:-:|
| 0  | 0  | 0 |
| 0  | 1  | 0 |
| 1  | 0  | 0 |
| 1  | 1  | 1 | 

A single perceptron can be used to learn the AND function. The perceptron you find in `learningAND.py` predicts `y` by calculating `activation(w1*x1 + w2*x2 + b)`. All the weights are initialized to some random value between -1 and 1 and updated everytime some prediction isn't correct (`output != y`). The activation function used is the [binary step](https://en.wikipedia.org/wiki/Heaviside_step_function). The program is ran 10 times with new random weights everytime, and a plot with the percent error in function of the iterations is drawn for each of them. 

## Learning OR

| x1 | x2 | y |
|:--:|:--:|:-:|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 1 | 

`learningOR.py` is exactly the same program as `learningAND.py`, the only difference is that the `and_examples` array was replaced with `or_examples`.

## Learning XOR 

| x1 | x2 | y |
|:--:|:--:|:-:|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 | 

`learningXOR.py` is exactly the same program as `learningAND.py`, the only difference is that the `and_examples` array was replaced with `xor_examples`. This program doesn't work (as expected), i.e. it is not able to learn weights so that the perceptron can correctly predict all the different XOR examples. This is because XOR is not [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).

To overcome this problem we have to create a multilayer perceptron, which is just a fancy way to say we have to add more perceptrons and connect them together. It is possible to learn the XOR function with just 2 perceptrons, but I used 3 in `learningXOR2.py` because I think it is more intuitive (and because it more straightforward to implement). In this program we have 2 perceptrons that are directly dependent on the input (just like we had before) and then we have a third perceptron, which is dependent on the output of the other two: 

```
zA = activation(wA1*x1 + wA2*x2 + bA)
zB = activation(wB1*x1 + wB2*x2 + bB)
z = activation(w1*zA + w2*zB + b)
```

The predictions are given by `z` and the weights (`wA1, wA2, bA, wB1, wB2, bB, w1, w2, b`) are updated using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) in order to decrease the mean square error (between `z` and `y`). You can see how the error changes with each iteration on the error_rate plots. The program is ran 10 times with the same initial random weights, but with ever increasing learning rates. When learning rate is less than 20, sometimes the program fails to learn the weights (it would learn them given enough iterations). When the learning is between 20 and 30 the program learns the XOR function without any problem. For learning rates above 30 the program also fails to learn the weights because the weight updates become too abrupt, which causes the mean square error to oscillate without ever reaching a minimum.

Although I am aware this is not the conventional way to program and train a multilayer perceptron, I decided to do this nonetheless to have have a clearer idea of how everything works under the hood. Nowadays people usually use a library such as tensorflow to do this kind of programs.`learningXOR3.py` creates the same multilayer perceptron and trains it the same way the `learningXOR2.py` does but using tensorflow and as you can see it has 1 order of magnitude less lines of code.

# Acknowledgement

A special thanks to [my dad](https://github.com/jvalentedeoliveira) for the precious help given with regards to the back propagation algorithm used on `learningXOR2.py`.

# License

This program is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
