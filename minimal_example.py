import numpy as np
import autograd.numpy as a_np
from autograd import elementwise_grad, grad, hessian
import pandas as pd
from scipy.optimize import minimize
import time

# A minimal example of issues with the estimator

#Gradient functions using autograd

def sigmoid(z):
    return 1/ (1 + a_np.exp(-z))

def zee(x_curr, datapoint):
    foo = a_np.dot(x_curr, datapoint)
    return foo

def logistic_gradient(x_curr, data):
    foo =  np.dot(data[:,:-1].T, (sigmoid(np.dot(x_curr, data[:,:-1].T)) - data[:,-1]))
    return foo

# Loss at a point
def loss(x_curr, point):
    d = point[:-1]
    loss = -(point[-1] * a_np.log(sigmoid(zee(x_curr, d))) + (1- point[-1]) * a_np.log(1-(sigmoid(zee(x_curr, d)))))
    return loss

# Total loss
def data_set_loss(x_curr, data):
    total = 0
    for point in data:
        total += loss(x_curr, point)
    return total

# Hessian gradient (we can't do math...)
total_hess = hessian(data_set_loss)

point_hessian = hessian(loss)

#Data, available in the repo and attached to the email
df = pd.read_csv("banknote.txt", header=None).values

# Add intercept
data = np.insert(df, -1, np.ones(df.shape[0]), axis=1)



#First, assert that Hessians are correct via Standard Newton's method

#Newton CG -- minimal value of ~24.94
x_curr = np.zeros(data.shape[1] - 1)
print((minimize(fun=data_set_loss, x0=x_curr, method="Newton-CG", jac=logistic_gradient, args=(data), options={'disp':True})))

time.sleep(1)
print("Scipy says minimum around 24.94")

# Now, do standard Newton's step, calculating the inverse (Not using the estimator)
print()
print("Now, calculate loss using standard Newton's method, without an estimator (do the matrix inversion)")
while(data_set_loss(x_curr, data) > 24.95):
    x_curr -= np.dot(np.linalg.inv(total_hess(x_curr, data)), logistic_gradient(x_curr, data))
    print("Loss =", data_set_loss(x_curr, data))


# Now comes our issues
time.sleep(.5)
print(data_set_loss(x_curr, data))
print("Now our issues:")
print("Norm of Hessian at solution =", np.linalg.norm(total_hess(x_curr, data)))
print("Norm of Hessian at solution is greater than 1 =", np.linalg.norm(total_hess(x_curr, data)) > 1)

# Because Norm is greater than 1, Equation 2, A^{-1} = \sum_i (I - A)^i doesn't necessarily hold.

#Thus, when we try to use the estimator (even just \sum_i (I-\hess(f))^i (even at the solution), it goes crazy

print("Using Estimator \\sum_i (I-\hess(f))^i")


print("Inverse of true hessian at solution:")
print(np.linalg.inv(total_hess(x_curr, data)))
# Start with zeros
total_estimator = np.zeros(total_hess(x_curr, data).shape)
# This is (I - \hess f)
term = np.eye(total_estimator.shape[0]) - total_hess(x_curr, data)
# Raising to the 0 gives I, and this produces the estimator of the Hessian as given in Eq. 2
for i in range(10):
    total_estimator += np.linalg.matrix_power(term, i)

# Nowhere near close to the same, and the estimator will quickly grow to infinity with more terms
print("Estimator at solution", total_estimator)

# This is the same when we do the sampling estimator
print("Using estimator at a point, as in Algorithm 1, \\sum_i (I-\hess(f(p)))^i")
total_estimator = np.zeros(total_estimator.shape)

for i in range(50):
    rand_idx = np.random.choice(data.shape[0])
    term = np.eye(total_estimator.shape[0]) - point_hessian(x_curr, data[rand_idx])
    total_estimator += np.linalg.matrix_power(term, i)

print("Estimator at solution using indivdual points as samples", total_estimator)


# Is this an issue with our dataset, or another misunderstanding of the algorithm? Thanks!