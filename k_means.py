#%%
import random
import numpy as np
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, clusters):
        self.clusters = clusters
        self.loss_history = []


    def fit(self, x, verbose=False):
        # Initialize cluster centroids
        self.centroids = np.array([x[np.random.randint(0, x.shape[0])]])
        for i in range(self.clusters - 1):
            self.centroids = np.append(self.centroids, np.array([x[np.random.randint(0, x.shape[0])]]), axis=0)
        
        # add column to show which centroid it belongs to
        m, n = x.shape
        temp = np.zeros((m, n+1))
        temp[:, :n] = x
        self.x = temp
        
        converged = False
        convergence_window = 1e-5
        epoch = 0

        while not converged:
            epoch += 1
            self.__closest_centroid(self.x)
            self.__update_centroid(self.x)

            loss = self.distortion_loss(self.x)

            if len(self.loss_history) > 0 and loss >= self.loss_history[-1]:
                if loss > self.loss_history[-1]:
                    print("Hmm?? Loss should not increase at all!!")
                converged = True
            self.loss_history.append(loss)
            
            if verbose:
                print(f"Epoch {epoch} log loss: {loss}")               

         
    def __closest_centroid(self, x):
        # Update each data point's closest centroid
        for i in range(x.shape[0]):
            x[i, -1] = np.argmin(np.linalg.norm(self.centroids - x[i, :-1], axis=1))

    def __update_centroid(self, x):
        # Update the centroids to the average position of data points
        
        for i in range(self.centroids.shape[0]):
            data_points = x[np.where(x[:, -1] == i)]
            self.centroids[i] = np.mean(data_points[:, :-1], axis=0)
        
    def distortion_loss(self, x):
        dist = np.copy(x[:, :-1])
        for i in range(dist.shape[0]):
            dist[i] = dist[i] - self.centroids[int(x[i, -1])]
        return np.mean(np.linalg.norm(dist, axis=1))


if __name__ == "__main__":

    NUMBER_OF_DATA = 50

    # random data
    x = np.random.random((NUMBER_OF_DATA, 2))
    x[:NUMBER_OF_DATA // 2, 0] = x[:NUMBER_OF_DATA // 2, 0] + 0.7
    x[NUMBER_OF_DATA // 2:, 1] = x[NUMBER_OF_DATA // 2:, 1] + 0.7

    plt.scatter(x[:, 0], x[:, 1])

    model = KMeans(2)
    model.fit(x, verbose=True)

    fig2 = plt.figure()
    plt.plot([x+1 for x in range(len(model.loss_history))], model.loss_history)

    
    colors = ['b', 'r', 'g', 'y']
    fig3 = plt.figure()
    for i in range(model.centroids.shape[0]):
        data = model.x[np.where(model.x[:, -1] == i)]
        plt.scatter(data[:, 0], data[:, 1], c=colors[i])
        plt.scatter(model.centroids[i][0], model.centroids[i][1], c=colors[i], marker='x')
    

# %%
