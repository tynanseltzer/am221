# We say that data is the standard format rows x columns, with last column being
# the class, and 2nd to last ones for the intercept

# Loss must be compatible with autograd
import autograd.numpy as a_np
from autograd import hessian
import numpy as np
import pandas as pd
from lissa import lissa
import time


# Helper functions
def sigmoid(z):
    return 1/ (1 + a_np.exp(-z))

def zee(x_curr, datapoint):
    foo = a_np.dot(datapoint, x_curr)
    return foo
# Loss should take current and data
# Loss at a point
def loss(x_curr, point):
    d = point[:-1]
    loss = -(point[-1] * a_np.log(sigmoid(zee(x_curr, d))) + (1- point[-1]) * a_np.log(1-(sigmoid(zee(x_curr, d)))))
    return loss

def data_set_loss(x_curr, data):
    total = 0
    for point in data:
        total += loss(x_curr, point)
    return total

#Without autograd
def logistic_gradient(x_curr, data):
    foo =  np.dot(data[:,:-1].T, (sigmoid(np.dot(x_curr, data[:,:-1].T)) - data[:,-1]))
    return foo

def logistic_hessian(x_curr, data):
    l = np.atleast_2d(sigmoid(zee(x_curr, data[:,:-1])))
    r = np.atleast_2d((1 - sigmoid(zee(x_curr, data[:,:-1]))))
    D = l.T.dot((r))
    D = np.diag(np.diag(D))
    return data[:,:-1].T.dot(D).dot((data[:,:-1]))





df = pd.read_csv("banknote.txt", header=None).values
data = np.insert(df, -1, np.ones(df.shape[0]), axis=1)

x_start = np.zeros(data.shape[1] - 1)

# print(time.time())
# print(lissa(x_start=x_start, loss=data_set_loss, data=data, S1=1, S2=15000, tol=.01, grad=logistic_gradient, hess=logistic_hessian))
# print(time.time())
# This uses BFGS
# (minimize(fun=old_loss, x0=np.zeros(data.shape[1] - 1), method="Newton-CG", jac=logistic_gradient, args=(data), options={'disp':True}))
# result = (minimize(fun=old_loss, x0=np.zeros(data.shape[1] - 1), args=(data), options={'disp':True}))
# print(result['x'])

# Let's use Newton-CG
# (minimize(fun=old_loss, x0=np.zeros(data.shape[1] - 1), method="Newton-CG", jac=logistic_gradient, args=(data), options={'disp':True}))

# Normal newton
x_curr = x_start
print(time.time())
while(data_set_loss(x_curr, data) > 24.95):
    x_curr -= np.dot(np.linalg.inv(logistic_hessian(x_curr, data)), logistic_gradient(x_curr, data))
    print("Loss =", data_set_loss(x_curr, data))
print(time.time())