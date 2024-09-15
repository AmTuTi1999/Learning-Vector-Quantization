import numpy as np
from LVQ.lvq_base import LVQBase
import logging
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from LVQ.init import initialize_prototypes, initialize_weights, weightedL2, sigmoid

logging.basicConfig(level=logging.DEBUG, format='%(filename)s:%(lineno)d - %(message)s')

class RLVQ(LVQBase):
    def __init__(
        self, 
        num_prototypes_per_class, 
        initialization_type = 'mean', 
        learning_rate = 0.05,
        weight_learning_rate = 0.01,
        max_iter = 100, 
    ):

        self.max_iter = max_iter 
        self.num_prototypes = num_prototypes_per_class
        self.initialization_type = initialization_type
        self.alpha = learning_rate
        self.eps = weight_learning_rate
        super(RLVQ).__init__()
        
    def update_weights(self, data, labels, prototypes, protolabels, weights):
        """
        Update the feature weights based on the distances between data points and their nearest prototypes.

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features).
            labels (np.ndarray): True labels for each data point.
            prototypes (np.ndarray): Prototype vectors of shape (n_prototypes, n_features).
            protolabels (np.ndarray): Labels corresponding to each prototype.
            weights (np.ndarray): Current feature weights of shape (n_features,).

        Returns:
            np.ndarray: Updated feature weights.
        """
        for xi, xlabel in zip(data, labels):
            distances = np.array([weightedL2(xi, p, weights) for p in prototypes])
            nearest_index = np.argmin(distances)
            nearest_prototype = prototypes[nearest_index]
            weight_update = self.eps * weightedL2(xi, nearest_prototype, weights) 
            if xlabel == protolabels[nearest_index]:
                weights -= weight_update
            else:
                weights += weight_update
            weights = np.clip(weights, a_min=0, a_max=None)
            weights /= np.sum(weights)
        return weights
    
    def cost(self, data, labels, prototypes, proto_labels, weights):
        """
        Compute the cost based on weighted distances between data points and prototypes.

        Args:
            data (np.ndarray): Array of input data points.
            labels (np.ndarray): Array of labels corresponding to the input data.
            prototypes (np.ndarray): Array of prototype vectors.
            proto_labels (np.ndarray): Array of labels corresponding to the prototypes.
            weight (np.ndarray): Weight vector to calculate weighted distances.

        Returns:
            float: The total cost, which is the sum of the sigmoid-transformed relative distances.
        """
        costs = []
        for i in range(len(data)):
            xi = data[i]
            x_label = labels[i]
            dist_a = np.array([weightedL2(xi, prototypes[j], weights) 
                            for j in range(len(prototypes)) if x_label == proto_labels[j]])
            d_a = dist_a.min()
            dist_b = np.array([weightedL2(xi, prototypes[j], weights) 
                            for j in range(len(prototypes)) if x_label != proto_labels[j]])
            d_b = dist_b.min()
            rel_dist = (d_a - d_b) / (d_a + d_b)
            costs.append(sigmoid(rel_dist).flatten())
        total_cost = np.sum(np.array(costs))
        return total_cost
    
    def update_prototypes(self, data, labels, prototypes, protolabels, weights):
        """
        Update the prototypes based on the distances between data points and their nearest prototypes.

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features).
            labels (np.ndarray): True labels for each data point.
            protolabels (np.ndarray): Labels corresponding to each prototype.
            prototypes (np.ndarray): Prototype vectors of shape (n_prototypes, n_features).
            weights (np.ndarray): Feature weights of shape (n_features,).

        Returns:
            np.ndarray: Updated prototypes.
        """
        for xi, xlabel in zip(data, labels):
            distances = np.array([weightedL2(xi, p, weights) for p in prototypes])
            nearest_index = np.argmin(distances)
            nearest_prototype = prototypes[nearest_index]
            prototype_update = self.alpha * (xi - nearest_prototype)
            if xlabel == protolabels[nearest_index]:
                prototypes[nearest_index] += prototype_update
            else:
                prototypes[nearest_index] -= prototype_update
        return prototypes

    def fit(self, data, labels, decay_scheme=True, show_plot=False) -> tuple:
        """
        Fit the model to the training data by iteratively updating weights and prototypes.

        Args:
            data (np.ndarray): Training data of shape (n_samples, n_features).
            labels (np.ndarray): True labels for each data point.
            decay_scheme (bool, optional): Whether to use decay scheme for epsilon and alpha. Defaults to True.

        Returns:
            tuple: Updated prototypes, prototype labels, and feature weights.
        """
        # Initialize prototypes and weights
        self._protolabels, self._prototypes = initialize_prototypes(
            data, labels, initialization_type=self.initialization_type, num_prototypes_per_class=self.num_prototypes
        )
        self._weights = initialize_weights(data)
        loss = []
        for iter in tqdm(range(self.max_iter), desc="Training Progress"):
            if decay_scheme:
                self.eps *= math.exp(-iter / self.max_iter)
                self.alpha *= math.exp(-iter / self.max_iter)
            self._weights = self.update_weights(data, labels, self._prototypes, self._protolabels, self._weights)
            self._prototypes = self.update_prototypes(data, labels, self._prototypes, self._protolabels, self._weights)
            if iter % 10 == 0:
                predicted = self.predict(data)
                accuracy = (np.array(predicted) == labels).mean() * 100
                loss = self.cost(data, labels, self._prototypes, self._protolabels, )
                logging.info(f'Epoch {iter:04d} - Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}')
        logging.info("Training finished")

        if show_plot:
            plt.plot(loss)
            plt.ylabel('log likelihood ratio')
            plt.xlabel(' number of iterations')





    


 
    





   