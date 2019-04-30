# We say that data is the standard format rows x columns, with last column being
# the class, and 2nd to last ones for the intercept

# grad(point) gives gradient
# hess(point) gives hessian at point
import numpy as np
import pandas as pd
import autograd.numpy as a_np
from autograd import elementwise_grad, grad, hessian
from scipy.optimize import minimize

# Trying out various line search methods. Paper seems to suggest just using 1 as coefficient of step, our convergence
# gets much slower when we do that
from scipy.optimize import line_search
from scipy.optimize.optimize import _line_search_wolfe12

import time

# An implementation of Algorithm 1 from Second-Order Stochastic Optimization for Machine Learning
# in Linear Time

#Note that we don't use a first order method as described, but right now, we're not seeing fast convergence near the
#optimum, so a FO method wouldn't help?
def lissa(x_start, grad, hess, data, NUM_ITERS, S1, S2):
    x_curr = x_start
    for t in range(1, NUM_ITERS + 1):
        # This holds the S1 unbiased estimators of Hessian inverse
        step_list = []
        for i in range(1, S1 + 1):
            step_list.append((grad(x_curr, data)))
            # print(grad(x_curr, data))
            # print(logistic_gradient(x_curr, data))
            # The points we use to get the depth of the Hessian
            idxs = np.random.choice(data.shape[0], size=S2)
            # Cut the last column (it's the class) (CURRENTLY CUT IN LOSS FUNCTION)
            #points = (data[idxs, :-1])
            points = (data[idxs])
            np.random.shuffle(points)
            for p in points:
                # Note that we overwrite, as opposed to storing in an array and only using last element
                # as in the paper.

                step_list[-1] = grad(x_curr, data) + np.dot((np.eye(data.shape[1]-1) - hess(x_curr, p)), step_list[-1])
                #print(step_list[-1])
            # We believe there to be a typo in the algorithm, and that you add the gradient once after the iterated
            # Hessian Estimation instead of multiplying it in at every step (as notation seems to hint in the paper)
            # thoughts?
            #step_list[-1] += grad(x_curr, data)
            print(step_list[-1])
            true_step = (np.dot((np.linalg.inv(full_hess(x_curr, data))), grad(x_curr, data)))
        total = sum(step_list)
        avg = total / S1
        #x_curr -= line_search(loss, logistic_gradient, xk= x_curr, pk=-avg, args=([data]), maxiter=1000)[0] * avg
        #x_curr -= _line_search_wolfe12(lambda x: loss(x, data), lambda x: logistic_gradient(x, data), xk=x_curr, pk=-avg, gfk=None, old_fval=None, old_old_fval=None)[0] * avg
        x_curr -= true_step
        if t % 1 == 0:
            print(t)
            print(data_set_loss(x_curr, data))
    return x_curr

def sigmoid(z):
    return 1/ (1 + a_np.exp(-z))

def zee(x_curr, datapoint):
    foo = a_np.dot(x_curr, datapoint)
    return foo

def sigmoid2(z):
    return 1/ (1 + np.exp(-z))

def zee2(x_curr, datapoint):
    foo = np.dot(x_curr, datapoint)
    return foo

# From https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function
def logistic_gradient(x_curr, data):
    foo =  np.dot(data[:,:-1].T, (sigmoid(np.dot(x_curr, data[:,:-1].T)) - data[:,-1]))/ data.shape[0]
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

def data_set_grad(x_curr, data):
    g = 0
    for point in data:
        g += grad_loss(x_curr, point)
    return g / data.shape[0]

def data_set_loss(x_curr, data):
    total = 0
    for point in data:
        total += loss(x_curr, point)
    return total

def old_loss(x_curr, data):
    loss = 0
    for i in range(data.shape[0]):
        d = data[i][:-1]
        loss += -(data[i][-1] * np.log(sigmoid2(zee2(x_curr, d))) + (1- data[i][-1]) * np.log(1-(sigmoid2(zee2(x_curr, d)))))
    return loss


full_hess = hessian(data_set_loss)

# Data in the repo
df = pd.read_csv("banknote.txt", header=None).values
# for row in df:
#     if row[-1] == 0:
#         row[-1] = -1

# df[df[:,-1] == 0][:,-1] = -1

# Add the intercept
data = np.insert(df, -1, np.ones(df.shape[0]), axis=1)

print(time.time())
# # This uses BFGS
# result = (minimize(fun=data_set_loss, x0=np.zeros(data.shape[1] - 1), args=(data), options={'disp':True}))
# print(result['x'])
# print(time.time())
# Let's use Newton-CG
(minimize(fun=old_loss, x0=np.zeros(data.shape[1] - 1), method="Newton-CG", jac=data_set_grad, args=(data), options={'disp':True}))
# print(time.time())
# Other methods we could compare to??

# Seems to converge to minimum of ~24.9 after ~1000 iterations, worse without line search
x_start = np.array([-4.06045034, -2.2588729, -2.7719268, -0.21613608, 4.08818242])
#print(lissa(x_start=x_start, grad=data_set_grad, hess=hess_loss, data=data, NUM_ITERS=1000, S1=1, S2=0))