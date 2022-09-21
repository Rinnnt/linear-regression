#%%

import random
import math
import matplotlib.pyplot as plt


class LinReg:
    def __init__(self, weight, bias, learning_rate):
        self.weight = weight
        self.bias = bias
        self.learning_rate = learning_rate
        self.mse_history = []
    
    def fit(self, x, y, verbose=False):
        convergence_window = 10 ** (-7)
        converged = False
        epoch = 0

        while not converged:
            epoch += 1
            new_weight, new_bias = self.__gradient_descent(x, y)
            if abs(new_weight - self.weight) < convergence_window and abs(new_bias - self.bias) < convergence_window:
                converged = True
            
            self.weight, self.bias = new_weight, new_bias

            if verbose and epoch % 10000 == 0:
                mse = self.__mse(self.predict(x), y)
                self.mse_history.append(mse)
                print(f"Epoch {epoch} MSE: {mse}")
        
    def __gradient_descent(self, x, y):
        preds = self.predict(x)
        new_weight = self.weight - self.learning_rate * ( sum([(pred - yi) * xi for xi, pred, yi in zip(x, preds, y)]) / len(preds) )
        new_bias = self.bias - self.learning_rate * ( sum([ pred - yi for pred, yi in zip(preds, y)]) / len(preds) )
        return new_weight, new_bias
        

    def __mse(self, preds, y):
        return sum([math.pow(pred - yi, 2) for pred, yi in zip(preds, y)]) / (2 * len(preds))

    def predict(self, x):
        return [self.weight * xi + self.bias for xi in x]


if __name__ == "__main__":
    # Correct Underlying Function 
    # y = wx + b
    # Include noise
    # xi is ith value of x, similar for ni

    WEIGHT = 3
    BIAS = 5
    NOISE_LEVEL = 1
    TRAINING_EXAMPLES = 20

    x = [random.random() * 10 for i in range(TRAINING_EXAMPLES)]
    noise = [(random.random() - 0.5) * NOISE_LEVEL for i in range(TRAINING_EXAMPLES)]
    y = [ (WEIGHT * xi + BIAS) + ni for xi, ni in zip(x, noise)]
    

    # Fit model 
    linreg_model = LinReg(0, 0, 0.0003)
    linreg_model.fit(x, y, verbose=True)
    print(linreg_model.weight)
    print(linreg_model.bias)

    # Visualize
    fig1 = plt.figure("Learning Curve")
    plt.plot([(x+1) * 10000 for x in range(len(linreg_model.mse_history))], linreg_model.mse_history)

    fig2 = plt.figure("Fitted Line")
    plt.plot(x, y, 'bo')
    plt.plot(x, linreg_model.predict(x))

# %%
