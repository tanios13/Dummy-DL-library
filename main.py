import utils
import numpy as np

if __name__ == '__main__':
    # define a dummy model
    data = np.random.normal()
    w1 = np.random.normal()
    b1 = np.random.normal()
    # used later
    w2 = np.random.normal()
    b2 = np.random.normal()
    w3 = np.random.normal()
    b3 = np.random.normal()

    model_operations = [
        lambda x: utils.multiply(x, w1),
        lambda x: utils.add(x, b1),
        utils.relu,
        lambda x: utils.multiply(x, w2),
        lambda x: utils.add(x, b2),
        utils.relu,
        lambda x: utils.multiply(x, w3),
    ]

    # we predict the output
    output, grad_funcs = utils.predict(ops_list=model_operations, _input=data)

    # get all the gradients
    print(utils.compute_grads(grad_funcs))