import numpy as np
from LVQ.lvq_base import LVQBase
import logging
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from LVQ.init import initialize_prototypes, initialize_weights, weightedL2, sigmoid, sigmoid_prime

logging.basicConfig(level=logging.DEBUG, format='%(filename)s:%(lineno)d - %(message)s')

class GLVQ(LVQBase):
    def __init__(
            self, 
            num_prototypes_per_class: int = 1, 
            initialization_type = 'mean',  
            learning_rate = 0.01,
            max_iter = 100, 
        ):
 
        self.num_prototypes = num_prototypes_per_class
        self.initialization_type = initialization_type
        self.alpha_zero = learning_rate
        self.max_iter = max_iter
        super(GLVQ).__init__()
    
    def update_prototypes(self, data, labels, prototypes, proto_labels):
        """
        Update the prototypes based on the distance between data points and their corresponding prototypes.

        Args:
            data (np.ndarray): Array of input data points.
            labels (np.ndarray): Array of labels corresponding to the input data.
            prototypes (np.ndarray): Array of prototype vectors to be updated.
            proto_labels (np.ndarray): Array of labels corresponding to the prototypes.

        Returns:
            np.ndarray: Updated prototypes.
        """
        for i in range(len(data)):
            xi = data[i]
            x_label = labels[i]
            dist_a = np.array([np.linalg.norm(xi - prototypes[j]) for j in range(len(prototypes)) if x_label == proto_labels[j]])
            dist_b = np.array([np.linalg.norm(xi - prototypes[j]) for j in range(len(prototypes)) if x_label != proto_labels[j]])
            d_a = dist_a.min()
            index_a = np.argmin(dist_a)
            d_b = dist_b.min()
            index_b = np.argmin(dist_b)
            rel_dist = (d_a - d_b) / (d_a + d_b)
            f = sigmoid(rel_dist)
            adjustment = self.alpha * (f * (1 - f)) / (d_a + d_b) ** 2
            prototypes[index_a] += adjustment * (d_b * (xi - prototypes[index_a]))
            prototypes[index_b] -= adjustment * (d_a * (xi - prototypes[index_b]))
        return prototypes
    
    def cost(self, data, labels, prototypes, proto_labels):
        """
        Compute the cost function based on the relative distances between data points 
        and their corresponding prototypes, using a sigmoid-based distance transformation.

        Args:
            data (np.ndarray): Array of input data points.
            labels (np.ndarray): Array of labels corresponding to the input data.
            prototypes (np.ndarray): Array of prototype vectors.
            proto_labels (np.ndarray): Array of labels corresponding to the prototypes.

        Returns:
            float: Normalized cost calculated using a sigmoid function over the relative distances.
        """
        d_a = np.array([np.min([np.linalg.norm(data[i] - prototypes[j]) 
                                for j in range(len(prototypes)) if labels[i] == proto_labels[j]]) 
                        for i in range(len(data))])
        d_b = np.array([np.min([np.linalg.norm(data[i] - prototypes[j]) 
                                for j in range(len(prototypes)) if labels[i] != proto_labels[j]]) 
                        for i in range(len(data))])
        rel_dist = (d_a - d_b) / (d_a + d_b)
        loss = np.sum(sigmoid(rel_dist))
        return loss / len(data)


    def fit(self, train_data, train_labels, decay_scheme = True, plot_loss = False):
        logging.info(f"Initializing prototypes with {self.initialization_type}")  
        prototype_labels, prototypes = initialize_prototypes(
            train_data, train_labels, initialization_type=self.initialization_type, num_prototypes_per_class=self.num_prototypes
        )
        prototypes = prototypes.astype(float)
        self._prototypes, self._protolabels = prototypes, prototype_labels
        loss = []
        for iter in tqdm(range(self.max_iter), desc="Training Progress"):
            if decay_scheme == True:
                self.alpha = self.alpha*(math.exp(-1*iter/self.max_iter))
                self._prototypes = self.update_prototypes(train_data, train_labels, self._prototypes, self._protolabels)
            self._prototypes = self.update_prototypes(train_data, train_labels, self._prototypes, self._protolabels)
            predicted = self.predict(train_data)
            error = self.cost(train_data, train_labels, self._prototypes, self._protolabels)
            loss.append(error)
            if iter % 10 == 0:
                predicted = self.predict(train_data)
                accuracy = (np.array(predicted) == train_labels).mean() * 100
                logging.info(f'Epoch {iter:04d} - Accuracy: {accuracy:.2f}%, Loss: {error:.4f}')
        if plot_loss == True:
            plt.plot(loss)
            plt.ylabel('log likelihood ratio')
            plt.xlabel(' number of iterations')

        logging.info("Training finished")
        

class GRLVQ(LVQBase):
    def __init__(
            self, 
            num_prototypes_per_class, 
            initialization_type = 'mean',  
            learning_rate = 0.01, 
            max_iter = 100,
            weight_update_learning_rate = 0.01,
        ):
 
        self.num_prototypes = num_prototypes_per_class
        self.initialization_type = initialization_type
        self.alpha = learning_rate
        self.max_iter = max_iter
        self.eps = weight_update_learning_rate
        super(GRLVQ).__init__()
    

    def update_prototypes(self, data, labels, prototypes, proto_labels, weight):  
        for i in range(len(data)):
            xi = data[i]
            x_label = labels[i]
            dist_a = np.array(
                [weightedL2(xi, prototypes[j],weight) for j in range(len(prototypes)) if x_label == proto_labels[j]]
            )
            d_a =dist_a.min()
            index_a = np.argmin(dist_a)
            dist_b = np.array(
                [weightedL2(xi, prototypes[j],weight) for j in range(len(prototypes)) if x_label != proto_labels[j]]
            )
            d_b = dist_b.min()
            index_b = np.argmin(dist_b)
            rel_dist = (d_a - d_b)/(d_a + d_b)
            f = sigmoid(rel_dist)
            prototypes[index_a] += self.alpha*(f*(1-f))*(np.divide(d_b, (d_a + d_b)**2))*(xi - prototypes[index_a])
            prototypes[index_b] -= self.alpha*(f*(1-f))*(np.divide(d_a, (d_a + d_b)**2))*(xi - prototypes[index_b])
            weight  -= self.eps*sigmoid_prime((np.divide(d_b, (d_a + d_b)**2))*(xi - prototypes[index_a])**2 - (np.divide(d_a, (d_a + d_b)**2))*(xi - prototypes[index_b])**2)
            weight = weight.clip(min = 0)
            weight = weight/weight.sum()         
        return prototypes, weight

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

      
    def fit(self, train_data, train_labels, decay_scheme = True, plot_loss = False):
        self._protolabels, self._prototypes = initialize_prototypes(
            train_data, train_labels, initialization_type=self.initialization_type, num_prototypes_per_class=self.num_prototypes
        )
        self._weights = initialize_weights(train_data)
        loss = []
        for iter in tqdm(range(self.max_iter), desc="Training Progress"):
            if decay_scheme :
                self.alpha = self.alpha*(math.exp(-1*iter/self.max_iter))
                self.eps = self.eps*(math.exp(-1*iter/self.max_iter))
            self._prototypes, self._weights = self.update_prototypes(train_data, train_labels, self._prototypes, self._protolabels, self._weights)
            error = self.cost(train_data, train_labels, self._prototypes, self._protolabels, self._weights)
            loss.append(error)
            if iter % 10 == 0:
                predicted = self.predict(train_data)
                accuracy = (np.array(predicted) == train_labels).mean() * 100
                logging.info(f'Epoch {iter:04d} - Accuracy: {accuracy:.2f}%, Loss: {error:.4f}')
        if plot_loss == True:
            plt.plot(loss)
            plt.ylabel('log likelihood ratio')
            plt.xlabel(' number of iterations')

        logging.info("Training finished")

    
        