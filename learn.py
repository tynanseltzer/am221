import numpy as np
import autograd.numpy as a_np
from autograd import elementwise_grad, grad, hessian
import pandas as pd

df = pd.read_csv("banknote.txt", header=None).values
# for row in df:
#     if row[-1] == 0:
#         row[-1] = -1

# df[df[:,-1] == 0][:,-1] = -1

# Add the intercept
data = np.insert(df, -1, np.ones(df.shape[0]), axis=1)





def sigmoid(z):
    return 1/ (1 + a_np.exp(-z))

def zee(x_curr, datapoint):
    foo = a_np.dot(x_curr, datapoint)
    return foo

# From https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function
def logistic_gradient(x_curr, data):
    z = zee(x_curr, data[:,:-1].T)
    inner = sigmoid(z - data[:,-1])
    foo =  np.dot(data[:,:-1].T, inner)/ data.shape[0]
    return foo

# def logistic_hessian(x_curr, data_point):
#     data_point = np.atleast_2d(data_point)
#     x_curr = np.atleast_2d(x_curr)
#     exes = np.dot(data_point.T, data_point)
#     sig_1 = sigmoid(zee(x_curr.T, data_point))
#     sig_2 = sigmoid(1 - zee(x_curr.T, data_point))
#
#     res = exes * (sig_1.dot(sig_2))
#     print("This is Hessian, and whether it is positive semi-definite", np.all(np.linalg.eigvals(res) > 0), res)
#     return res

# Loss at a point
def loss(x_curr, point):
    d = point[:-1]
    loss = -(point[-1] * a_np.log(sigmoid(zee(x_curr, d))) + (1- point[-1]) * a_np.log(1-(sigmoid(zee(x_curr, d)))))
    return loss


hess_loss = hessian(loss)
grad_loss = grad(loss)

# def data_set_grad(x_curr, data):
#     g = 0
#     for point in data:
#         g += grad_loss(x_curr, point)
#     return g / data.shape[0]


def data_set_loss(x_curr, data):
    total = 0
    for point in data:
        total += loss(x_curr, point)
    return total

data_set_grad = grad(data_set_loss)

def logistic_gradient(x_curr, data):
    foo =  np.dot(data[:,:-1].T, (sigmoid(np.dot(x_curr, data[:,:-1].T)) - data[:,-1]))
    return foo


def logistic_hessian(x_curr, data_point):
    data_point = np.atleast_2d(data_point)
    x_curr = np.atleast_2d(x_curr)
    exes = np.dot(data_point.T, data_point)
    sig_1 = sigmoid(zee(x_curr.T, data_point))
    sig_2 = sigmoid(1 - zee(x_curr.T, data_point))

    res = np.dot(exes,(sig_1.dot(sig_2)))
    # print("This is Hessian, and whether it is positive semi-definite", np.all(np.linalg.eigvals(res) > 0), res)
    return res

def full_hess(x_curr, data):
    tot = 0
    for p in data:
        p = p[:-1]
        tot += logistic_hessian(x_curr, p)
    return tot / data.shape[0]

full_hess2 = hessian(data_set_loss)

x_curr = np.array([-4.06045034, -2.2588729, -2.7719268, -0.21613608, 4.08818242])
x_curr = np.zeros(5)

# Data in the repo
df = pd.read_csv("banknote.txt", header=None).values
# for row in df:
#     if row[-1] == 0:
#         row[-1] = -1

# df[df[:,-1] == 0][:,-1] = -1

# Add the intercept
data = np.insert(df, -1, np.ones(df.shape[0]), axis=1)


while True:
    x_curr = x_curr - np.dot(np.linalg.inv(full_hess2(x_curr, data)), logistic_gradient(x_curr, data))
    temp = (full_hess2(x_curr, data))
    print(np.linalg.inv(full_hess2(x_curr, data)))
    total_estim = np.zeros(temp.shape)
    estim = (np.eye(temp.shape[0]) - full_hess2(x_curr, data))
    for i in range(5):
        total_estim += estim ** i
    #print(np.dot(logistic_gradient(x_curr, data), total_estim))
    print("Estimated", total_estim)