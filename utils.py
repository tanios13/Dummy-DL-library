import numpy as np

def relu(x):
    output = x * (x > 0.)

    def grad_f(outgrad):
        d_dx = np.ones_like(x) * (x > 0.)
        return d_dx * outgrad

    return output, grad_f

def multiply(x, w):
    output = x * w

    def grad_f(outgrad):
        d_dx = w
        d_dw = x
        return d_dx * outgrad, d_dw * outgrad

    return output, grad_f


def add(x, b):
    output = x + b

    def grad_f(outgrad):
        d_dx = 1.
        d_db = 1.
        return d_dx * outgrad, d_db * outgrad

    return output, grad_f

def predict(ops_list, _input):
    grad_funcs = []
    x = _input
    for op in ops_list:  # [multiply, add]
        # apply operation, e.g. multiply, add
        x, grad_f = op(x)
        # save gradient functions
        grad_funcs.append(grad_f)
    return x, grad_funcs


def compute_grads(grad_funcs):
    grad_vals = []
    grad_x = 1.
    # iterate grad functions from the end
    # e.g. [add, multiply]
    for grad_f in reversed(grad_funcs):
        grad = grad_f(outgrad=grad_x)
        has_multiple_outputs = isinstance(grad, tuple)
        if has_multiple_outputs:
            # Assume the 2nd param is a variable
            grad_x, grad_v = grad
            grad_vals.append(grad_v)
        else:
            grad_x = grad
            grad_vals.append(None)
    # return the gradient values
    # in the order of operations
    return list(reversed(grad_vals))