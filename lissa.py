# We say that data is the standard format rows x columns, with last colum being
# the class, and 2nd to last ones for the intercept
# grad(point) gives gradient
# hess(point) gives hessian at point
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import line_search
def lissa(x_start, grad, hess, data, NUM_ITERS, S1, S2):
    x_curr = x_start
    for t in range(1, NUM_ITERS + 1):
        step_list = []
        for i in range(1, S1 + 1):
            step_list.append(np.zeros(data.shape[1]-1))
            for j in range(1, S2 + 1):
                idx = np.random.choice(data.shape[0])
                p = data[idx][:-1]
                # Note that we overwrite, as opposed to storing in an array and only using last element
                # as in the paper.
                step_list[-1] = np.dot(step_list[-1].T,  (np.eye(data.shape[1]-1) - hess(x_curr, p)).T) + grad(x_curr, data)
        total = sum(step_list)
        avg = total / S1
        x_curr -= line_search(loss, logistic_gradient, xk= x_curr, pk=-avg, args=([data]))[0] * avg
        print(t)
        print(x_curr)
        print(loss(x_curr, data))
    return x_curr

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

def zee(x_curr, datapoint):
    foo = np.dot(x_curr, datapoint)
    return foo

# From https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function
def logistic_gradient(x_curr, data):
    foo =  np.dot(data[:,:-1].T, (sigmoid(np.dot(x_curr, data[:,:-1].T)) - data[:,-1]))/ data.shape[0]
    return foo

def logistic_hessian(x_curr, data_point):
    data_point = np.atleast_2d(data_point)
    x_curr = np.atleast_2d(x_curr)
    bar =  np.dot(data_point.T, data_point)*sigmoid(np.dot(x_curr, data_point.T))*(1-sigmoid(np.dot(x_curr, data_point.T)))
    return bar

def loss(x_curr, data):
    loss = 0
    for i in range(data.shape[0]):
        d = data[i][:-1]
        loss += -(data[i][-1] * np.log(sigmoid(zee(x_curr, d))) + (1- data[i][-1]) * np.log(1-(sigmoid(zee(x_curr, d)))))
    return loss

df = pd.read_csv("banknote.txt", header=None).values
data = np.insert(df, -1, np.ones(df.shape[0]), axis=1)
print(minimize(loss, np.zeros(data.shape[1] - 1), args=(data)))

print(data[0])
print(lissa([-5.0042507,  -3.02215979, -3.4448039,  -1.22954548,  2.04449799], logistic_gradient, logistic_hessian, data, 20000, 1000, 4))