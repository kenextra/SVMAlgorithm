import numpy as np


def confusion_matrix(y_actual, y_predicted):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(y_actual)):
        if y_actual[i] > 0:
            if y_actual[i] == y_predicted[i]:
                tp = tp + 1
            else:
                fn = fn + 1
        if y_actual[i] < 1:
            if y_actual[i] == y_predicted[i]:
                tn = tn + 1
            else:
                fp = fp + 1

    cm = [[tn, fp], [fn, tp]]
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sens = tp/(tp+fn)
    prec = tp/(tp+fp)
    f_score = (2*prec*sens)/(prec+sens)
    return cm, accuracy, f_score


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    numerator = np.linalg.norm(x-y)**2
    denominator = 2 * (sigma ** 2)
    return np.exp(-numerator / denominator)


class SVM(object):

    def __init__(self, kernel=linear_kernel, tol=1e-3, C=0.1,
                 max_passes=5, sigma=0.1):

        self.kernel = kernel
        self.tol = tol
        self.C = C
        self.max_passes = max_passes
        self.sigma = sigma
        self.model = dict()

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"kernel={self.kernel.__name__}, "
                f"tol={self.tol}, "
                f"C={self.C}, "
                f"max_passes={self.max_passes}, "
                f"sigma={self.sigma}"
                ")")

    def svmTrain(self, X, Y):
        # Data parameters
        m = X.shape[0]

        # Map 0 to -1
        Y = np.where(Y == 0, -1, 1)

        # Variables
        alphas = np.zeros((m, 1), dtype=float)
        b = 0.0
        E = np.zeros((m, 1), dtype=float)
        passes = 0

        # Pre-compute the kernel matrix
        if self.kernel.__name__ == 'linear_kernel':
            print(f'Pre-computing {self.kernel.__name__} matrix')
            K = X @ X.T

        elif self.kernel.__name__ == 'gaussian_kernel':
            print(f'Pre-computing {self.kernel.__name__} matrix')
            X2 = np.sum(np.power(X, 2), axis=1).reshape(-1, 1)
            K = X2 + (X2.T - (2 * (X @ X.T)))
            K = np.power(self.kernel(1, 0, self.sigma), K)

        else:
            # Pre-compute the Kernel Matrix
            # The following can be slow due to lack of vectorization
            print(f'Pre-computing {self.kernel.__name__} matrix')
            K = np.zeros((m, m))

            for i in range(m):
                for j in range(m):
                    x1 = np.transpose(X[i, :])
                    x2 = np.transpose(X[j, :])
                    K[i, j] = self.kernel(x1, x2)
                    K[i, j] = K[j, i]

        print('Training...')
        print('This may take 1 to 2 minutes')

        while passes < self.max_passes:
            num_changed_alphas = 0

            for i in range(m):

                E[i] = b + np.sum(alphas * Y * K[:, i].reshape(-1, 1)) - Y[i]

                if (Y[i] * E[i] < -self.tol and alphas[i] < self.C) or (Y[i] * E[i] > self.tol and alphas[i] > 0):
                    j = np.random.randint(0, m)
                    while j == i:
                        # make sure i is not equal to j
                        j = np.random.randint(0, m)

                    E[j] = b + np.sum(alphas * Y *
                                      K[:, j].reshape(-1, 1)) - Y[j]

                    # Save old alphas
                    alpha_i_old = alphas[i, 0]
                    alpha_j_old = alphas[j, 0]

                    # Compute L and H by (10) or (11)
                    if Y[i] == Y[j]:
                        L = max(0, alphas[j] + alphas[i] - self.C)
                        H = min(self.C, alphas[j] + alphas[i])
                    else:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    if L == H:
                        # continue to next i
                        continue

                    # compute eta by (14)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        # continue to next i
                        continue

                    # compute and clip new value for alpha j using (12) and (15)
                    alphas[j] = alphas[j] - (Y[j] * (E[i] - E[j])) / eta

                    # Clip
                    alphas[j] = min(H, alphas[j])
                    alphas[j] = max(L, alphas[j])

                    # Check if change in alpha is significant
                    if np.abs(alphas[j] - alpha_j_old) < self.tol:
                        # continue to the next i
                        # replace anyway
                        alphas[j] = alpha_j_old
                        continue

                    # Determine value for alpha i using (16)
                    alphas[i] = alphas[i] + Y[i] * \
                        Y[j] * (alpha_j_old - alphas[j])

                    # Compute b1 and b2 using (17) and (18) respectively.
                    b1 = b - E[i] - Y[i] * (alphas[i] - alpha_i_old) * \
                        K[i, j] - Y[j] * (alphas[j] - alpha_j_old) * K[i, j]

                    b2 = b - E[j] - Y[i] * (alphas[i] - alpha_i_old) * \
                        K[i, j] - Y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                    # Compute b by (19).
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    num_changed_alphas = num_changed_alphas + 1

            if num_changed_alphas == 0:
                passes = passes + 1
            else:
                passes = 0

            print('.', end='', flush=True)

        print('\n DONE! ')

        # Save the model
        idx = alphas > 0
        self.model['X'] = X[idx.reshape(1, -1)[0], :]
        self.model['y'] = Y[idx.reshape(1, -1)[0]]
        self.model['kernelFunction'] = self.kernel
        self.model['b'] = b
        self.model['alphas'] = alphas[idx.reshape(1, -1)[0]]
        self.model['w'] = np.transpose(np.matmul(np.transpose(alphas * Y), X))
        # return model

    def svmPredict(self, X):
        if X.shape[1] == 1:
            X = np.transpose(X)

        # Dataset
        m = X.shape[0]
        p = np.zeros((m, 1))
        pred = np.zeros((m, 1))

        if self.model['kernelFunction'].__name__ == 'linear_kernel':
            p = X.dot(self.model['w']) + self.model['b']

        elif self.model['kernelFunction'].__name__ == 'gaussian_kernel':
            # Vectorized RBF Kernel
            # This is equivalent to computing the kernel
            # on every pair of examples
            X1 = np.sum(np.power(X, 2), axis=1).reshape(-1, 1)
            X2 = np.transpose(np.sum(np.power(self.model['X'], 2), axis=1))
            K = X1 + (X2.T - (2 * (X @ (self.model['X']).T)))
            K = np.power(self.model['kernelFunction'](1, 0, self.sigma), K)
            K = np.transpose(self.model['y']) * K
            K = np.transpose(self.model['alphas']) * K
            p = np.sum(K, axis=1)

        else:
            for i in range(m):
                prediction = 0
                for j in range(self.model['X'].shape[0]):
                    prediction = prediction + self.model['alphas'][j] \
                        * self.model['y'][j] * \
                        self.model['kernelFunction'](np.transpose(
                            X[i, :]), np.transpose(self.model['X'][j, :]))

                p[i] = prediction + self.model['b']

        # Convert predictions into 0 and 1
        pred[p >= 0] = 1
        return pred

    def predict(self, X):
        if X.shape[1] == 1:
            X = np.transpose(X)

        # Dataset
        m = X.shape[0]
        p = np.zeros((m, 1))
        pred = np.zeros((m, 1))

        if self.model['kernelFunction'].__name__ == 'linear_kernel':
            p = X.dot(self.model['w']) + self.model['b']

        elif self.model['kernelFunction'].__name__ == 'gaussian_kernel':
            # Vectorized RBF Kernel
            # This is equivalent to computing the kernel
            # on every pair of examples
            X1 = np.sum(np.power(X, 2), axis=1).reshape(-1, 1)
            X2 = np.transpose(np.sum(np.power(self.model['X'], 2), axis=1))
            K = X1 + (X2.T - (2 * (X @ (self.model['X']).T)))
            K = np.power(self.model['kernelFunction'](1, 0, self.sigma), K)
            K = np.transpose(self.model['y']) * K
            K = np.transpose(self.model['alphas']) * K
            p = np.sum(K, axis=1)

        else:
            for i in range(m):
                prediction = 0
                for j in range(self.model['X'].shape[0]):
                    prediction = prediction + self.model['alphas'][j] \
                        * self.model['y'][j] * \
                        self.model['kernelFunction'](np.transpose(
                            X[i, :]), np.transpose(self.model['X'][j, :]))

                p[i] = prediction + self.model['b']

        # Convert predictions into 0 and 1
        pred[p >= 0] = 1
        return pred
