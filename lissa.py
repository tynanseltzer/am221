# We say that data is the standard format rows x columns
# grad(point) gives gradient
# hess(point) gives hessian at point
import numpy as np
def lissa(x_start, grad, hess, data):
    NUM_ITERS = 100
    S1 = 20
    S2 = 10
    x_curr = x_start
    for t in range(1, NUM_ITERS + 1):
        step_list = []
        for i in range(1, S1 + 1):
            step_list.append(grad(x_curr))
            for j in range(1, S2 + 1):
                idx = np.random.choice(data.shape[0])
                p = data[idx]
                step_list[-1] = step_list[-1] * (np.eye() - hess(p)) + grad(x_curr)
        total = np.sum(step_list)
        avg = total / S1

        x_curr -= avg

