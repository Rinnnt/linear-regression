#%%
import numpy as np
import matplotlib.pyplot as plt

class AnomalyDetector:

    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def fit(self, x):
        self.weights = np.zeros((x.shape[1], 2))

        # Calculate mean
        self.weights[:, 0] = np.mean(x, axis=0)
        # Calculate Variance
        self.weights[:, 1] = np.mean(np.square(x - self.weights[:, 0]), axis=0)

    def __gaussian(self, x):
        exponent = - (np.square(x - self.weights[:, 0])) / (2 * self.weights[:, 1])
        coefficient = 1 / (np.sqrt(2 * np.pi) * np.sqrt(self.weights[:, 1]))
        return  coefficient * np.exp(exponent)

    def predict(self, x):
        probabilities = self.__gaussian(x)
        final_prob = np.prod(probabilities, axis=1)
        preds = final_prob < self.epsilon
        return preds.astype(int)

if __name__ == "__main__":

    NUMBER_OF_EXAMPLES = 50
    x = np.random.randn(NUMBER_OF_EXAMPLES, 3)
    x[:, 1] = (x[:, 1] * 2) + 6
    x[:, 2] = (x[:, 2] * 5) - 20

    model = AnomalyDetector(1e-5)
    model.fit(x)

    # 0 is normal, 1 is anomaly
    test = np.array([[1, 14, -20]])

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.scatter(x[:, 0], np.zeros((x[:, 0].shape)))
    plt.scatter(test[0, 0], 0, c='r', marker='x')

    plt.subplot(3, 1, 2)
    plt.scatter(x[:, 1], np.zeros((x[:, 1].shape)))
    plt.scatter(test[0, 1], 0, c='r', marker='x')
    
    plt.subplot(3, 1, 3)
    plt.scatter(x[:, 2], np.zeros((x[:, 2].shape)))
    plt.scatter(test[0, 2], 0, c='r', marker='x')
    print("ANOMALY" if model.predict(test)[0] else "normal")
#%%

