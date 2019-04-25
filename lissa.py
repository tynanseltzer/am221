# We say that data is the standard format rows x columns, with last column being
# the class, and 2nd to last ones for the intercept

# grad(point) gives gradient
# hess(point) gives hessian at point
import numpy as np
import pandas as pd
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
            step_list.append(np.zeros(data.shape[1] - 1))
            # The points we use to get the depth of the Hessian
            idxs = np.random.choice(data.shape[0], size=S2)
            # Cut the last column (it's the class)
            points = (data[idxs, :-1])
            np.random.shuffle(points)
            for p in points:
                # Note that we overwrite, as opposed to storing in an array and only using last element
                # as in the paper.
                step_list[-1] = np.dot((np.eye(data.shape[1]-1) - hess(x_curr, p)), step_list[-1])

            # We believe there to be a typo in the algorithm, and that you add the gradient once after the iterated
            # Hessian Estimation instead of multiplying it in at every step (as notation seems to hint in the paper)
            # thoughts?
            step_list[-1] += grad(x_curr, data)

        total = sum(step_list)
        avg = total / S1
        #x_curr -= line_search(loss, logistic_gradient, xk= x_curr, pk=-avg, args=([data]), maxiter=1000)[0] * avg
        #x_curr -= _line_search_wolfe12(lambda x: loss(x, data), lambda x: logistic_gradient(x, data), xk=x_curr, pk=-avg, gfk=None, old_fval=None, old_old_fval=None)[0] * avg
        x_curr -= avg
        if t % 50 == 0:
            print(t)
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

# Data in the repo
df = pd.read_csv("banknote.txt", header=None).values

# Add the intercept
data = np.insert(df, -1, np.ones(df.shape[0]), axis=1)

print(time.time())
# This uses BFGS
(minimize(fun=loss, x0=np.zeros(data.shape[1] - 1), args=(data), options={'disp':True}))
print(time.time())
# Let's use Newton-CG
(minimize(fun=loss, x0=np.zeros(data.shape[1] - 1), method="Newton-CG", jac=logistic_gradient, args=(data), options={'disp':True}))
print(time.time())
# Other methods we could compare to??

# Seems to converge to others after ~1000 iterations, worse without line search
print(lissa(x_start=np.zeros(data.shape[1] - 1), grad=logistic_gradient, hess=logistic_hessian, data=data, NUM_ITERS=1000, S1=1, S2=10000))