# We say that data is the standard format rows x columns, with last column being
# the class, and 2nd to last ones for the intercept

# grad(point) gives gradient
# hess(point) gives hessian
import numpy as np
from autograd import hessian
import autograd
import scipy.linalg


def lissa(loss, data, tol, S1=1, S2=15000, x_start=None, grad=None, hess=None):
    if x_start.all() != None:
        x_start = np.zeros(data.shape[1] - 1)
    if not grad:
        grad = autograd.grad(loss)
    if not hess:
        hess = hessian(loss)
    x_curr = x_start
    t = 0
    while(True):
        step_list = []
        for i in range(1, S1 + 1):

            h = hess(x_curr, data)
            step_list.append(np.zeros(h.shape))
            largest_eig = scipy.linalg.eigh(a=h, eigvals_only=True, eigvals=(h.shape[0] - 1, h.shape[0] - 1))
            term = np.eye(step_list[-1].shape[0]) - h / largest_eig
            for j in range(S2):
                step_list[-1] += np.linalg.matrix_power(term, j)
            step_list[-1] = np.dot(step_list[-1] / largest_eig, grad(x_curr, data))
        total = sum(step_list)
        avg = total / S1
        x_curr -= avg
        if np.sum(avg) < tol:
            break
        # if t % 1 == 0:
        #     print(t)
        #     print(loss(x_curr, data))
        t += 1
    return x_curr
