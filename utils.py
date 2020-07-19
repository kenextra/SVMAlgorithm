import math
import numpy as np
from SVM import SVM, gaussian_kernel


def parameter_search(X, y, Xval, yval, max_passes):
    C = 1
    sigma = 0.3
    param_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 5, 10, 30]
    init_error = math.inf

    for i in param_vec:
        c = i
        for j in param_vec:
            s = j
            model = SVM(kernel=gaussian_kernel, C=c,
                        sigma=s, max_passes=max_passes)
            model.svmTrain(X, y)
            predictions = model.svmPredict(Xval)
            error = np.mean(predictions != yval)
            if error < init_error:
                init_error = error
                C = c
                sigma = s
            print("C = ", c, "sigma = ", s, 'error = ', init_error)
    print(f"\nFinal parameters: C={C}, sigma={sigma}")
    return C, sigma
