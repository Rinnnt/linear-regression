#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



class LinRegMulti:

    def __init__(self, learning_rate):
        self.weights = np.array([])
        self.learning_rate = learning_rate
        self.mse_history = []

    def fit(self, x, y, verbose=False):
        
        # Include bias column if training data does not include it
        # x = self.__include_bias(x)

        self.weights = np.zeros((x.shape[1]))

        convergence_window = 10 ** (-7)
        converged = False
        epoch = 0
        
        while not converged:
            epoch += 1
            new_weights = self.__gradient_descent(x, y)
            
            if  np.linalg.norm(self.weights - new_weights) < convergence_window:
                converged = True

            self.weights = new_weights

            if verbose and epoch % 100 == 0:
                mse = self.__mse(self.predict(x), y)
                self.mse_history.append(mse)
                print(f"Epoch {epoch} MSE: {mse}")


    def __gradient_descent(self, x, y):
        delta = x.T.dot(self.predict(x) - y) / x.shape[0]
        new_weights = self.weights - self.learning_rate * delta
        return new_weights

    def __mse(self, preds, y):
        return np.sum(np.square(preds - y)) / (2 * len(preds))

    def predict(self, x):
        return x.dot(self.weights)

    def __include_bias(self, x):
        m, n = x.shape
        new = np.ones((m, n+1))
        new[:, 1:] = x
        return new
    


if __name__ == "__main__":
    # Correct Underlying Function 
    # y = w1x1 + w2x2 + b
    # Include noise

    WEIGHTS = np.array([1, 3, 5])
    NOISE_LEVEL = 1
    TRAINING_EXAMPLES = 200
    TEST_EXAMPLES = 50

    x_features = np.random.randn(TRAINING_EXAMPLES, 2) * 10
    x = np.ones((TRAINING_EXAMPLES, 3))
    x[:, 1:] = x_features

    noise = np.random.randn(TRAINING_EXAMPLES) * NOISE_LEVEL
    y = x.dot(WEIGHTS) + noise

    # test data
    test_x_features = np.random.randn(TEST_EXAMPLES, 2) * 10
    test_x = np.ones((TEST_EXAMPLES, 3))
    test_x[:, 1:] = test_x_features
    
    test_y = test_x.dot(WEIGHTS)

    # TRAIN MODEL
    model = LinRegMulti(0.001)
    model.fit(x, y, verbose=True)
    plt.plot([(x+1) * 100 for x in range(len(model.mse_history))], model.mse_history)
    print("MODEL WEIGHTS COMPARED TO ORIGINAL FUNCTION [1, 3, 5]")
    print(model.weights)

    test_mse = np.sum(np.square(model.predict(test_x) - test_y)) / (2 *len(test_x))
    print(f"TEST SET MSE: {test_mse}")
# %%
