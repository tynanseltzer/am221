# We say that data is the standard format rows x columns, with last colum being
# the class, and 2nd to last ones for the intercept
# grad(point) gives gradient
# hess(point) gives hessian at point
import numpy as np
import pandas as pd
def lissa(x_start, grad, hess, data, NUM_ITERS, S1, S2):
    x_curr = x_start
    for t in range(1, NUM_ITERS + 1):
        step_list = []
        for i in range(1, S1 + 1):
            step_list.append(grad(x_curr, data))
            for j in range(1, S2 + 1):
                idx = np.random.choice(data.shape[0])
                p = data[idx][:-1]
                # Note that we overwrite, as opposed to storing in an array and only using last element
                # as in the paper.
                step_list[-1] = np.dot(step_list[-1].T,  (np.eye(data.shape[1]-1) - hess(x_curr, p)).T) + grad(x_curr, data)
        total = np.sum(step_list)
        avg = total / S1

        x_curr -= avg
        print(x_curr)
    return x_curr

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

def zee(x_curr, datapoint):
    return np.dot(x_curr.T, da)

# From https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function
def logistic_gradient(x_curr, data):
    return np.dot(data[:,:-1].T, (sigmoid(np.dot(x_curr, data[:,:-1].T)) - data[:,-1]))/ data.shape[0]


def logistic_hessian(x_curr, data_point):
    return np.dot(data_point.T, data_point)*sigmoid(np.dot(x_curr, data_point.T))*(1-sigmoid(np.dot(x_curr, data_point.T)))

def loss(x_curr, data):
    loss = 0
    for i in range(data.shape[0]):
        loss += -(data[i][-1] * )

df = pd.read_csv("banknote.txt", header=None).values
data = np.insert(df, -2, np.zeros(df.shape[0]), axis=1)

print(lissa(np.zeros(data.shape[1] - 1), logistic_gradient, logistic_hessian, data, 10000, 10, 2))