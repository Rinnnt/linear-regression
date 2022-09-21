#%%
from pyexpat.errors import XML_ERROR_NOT_STANDALONE
import numpy as np
import matplotlib.pyplot as plt

class LogReg:

    def __init__(self, learning_rate, regularization_parameter=None, convergence_window=None):
        self.weights = np.array([])
        self.learing_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.convergence_window = convergence_window if convergence_window else 10 ** -5
        self.loss_history = []

    def fit(self, x, y, verbose=False):
        self.weights = np.zeros(x.shape[1])
        converged = False
        epoch = 0

        while not converged:
            epoch += 1
            new_weights = self.__gradient_descent(x, y)

            if np.linalg.norm(new_weights - self.weights) < self.convergence_window:
                converged = True
            
            self.weights = new_weights

            if verbose and epoch % 10000 == 0:
                loss = self.__log_loss(self.predict(x), y)
                self.loss_history.append(loss)
                print(f"Epoch {epoch} log loss: {loss}")


    def __gradient_descent(self, x, y):
        regularization_terms = np.zeros((x.shape[0]))
        if self.regularization_parameter:
            regularization_terms = (self.regularization_parameter/x.shape[0]) * self.weights
            regularization_terms[0] = 0
        
        delta = x.T.dot(self.predict(x) - y) / x.shape[0] + regularization_terms
        new_weights = self.weights - self.learing_rate * delta
        return new_weights


    def __log_loss(self, preds, y):
        regularization_terms = np.zeros((preds.shape[0]))
        if self.regularization_parameter:
            regularization_terms = (self.regularization_parameter/(2 * preds.shape[0])) * np.square(self.weights) 
            
            # No regularization on bias term
            regularization_terms[0] = 0
        return np.sum(((y) * -np.log(preds)) + ((1 - y) * -np.log(1-preds))) / (2 * preds.shape[0]) + np.sum(regularization_terms)

    def predict(self, x):
        # did not categorize into 0 or 1 
        # usually, if prediction > 0.5, categorize to 1
        return self.__sigmoid(x.dot(self.weights))

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    
    #
    TRAINING_EXAMPLES = 50
    DECISION_BOUNDARY = 12

    x_features = np.abs(np.random.randn(TRAINING_EXAMPLES, 1)) * 10
    x = np.ones((TRAINING_EXAMPLES, 2))
    x[:, 1:] = x_features

    y = (x_features > DECISION_BOUNDARY).astype(int).ravel()


    model = LogReg(0.01, regularization_parameter=0.1)
    model.fit(x, y, verbose=True)


    # Visualize sigmoid
    x_test_features = np.linspace(0, 30, 60).reshape(60, 1)
    x_test = np.ones((len(x_test_features), 2))
    x_test[:, 1:] = x_test_features

    preds = model.predict(x_test)
    
    plt.plot(x_test_features, preds)
    
    print(model.weights)
# %%
