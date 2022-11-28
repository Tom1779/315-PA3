import numpy as np

def perceptron(vector_table, labels, lr, weights):
    mistakes = [0]*20
    for iteration in range(20):
        for vector in range(len(vector_table)):
            y = np.dot(vector_table[vector], weights)
            if(y > 0):
                y = 1
            else:
                y = 0
            if(y != labels[vector]):
                mistakes[iteration] += 1
                weights = np.add(weights, lr * np.dot((labels[vector] - y), vector_table[vector]))


    print(weights)
    print(mistakes)