import numpy as np
import math
from LVQ.lvq_base import LVQBase
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from LVQ.init import initialize_prototypes

logging.basicConfig(level=logging.DEBUG, format='%(filename)s:%(lineno)d - %(message)s')

class SNPC(LVQBase):
    def __init__(
        self, 
        num_prototypes_per_class, 
        initialization_type = 'mean', 
        sigma = 1, 
        learning_rate = 0.05,
        max_iter = 100, 
    ):

        self.max_iter = max_iter
        self.num_prototypes = num_prototypes_per_class
        self.sigma = sigma
        self.initialization_type = initialization_type
        self.alpha = learning_rate
        super(SNPC).__init__()
    

    def inner_f(self, x, p):
        coef = -1/(2*(self.sigma *self.sigma))
        dist = (x -p)@(x- p).T
        return coef*dist

    def inner_derivative(self, x, p):
        coef = 1/(self.sigma *self.sigma)
        diff = (x -p) 
        return coef*diff
        
    def Pl(self, x, index, prototypes):
        inner = np.exp(np.array([self.inner_f(x, p) for p in  prototypes]))
        numerator = np.exp(np.array(self.inner_f(x, prototypes[index])))
        denominator = inner.sum()
        return numerator/(denominator) 

    def lst(self, x, x_label, prototypes, proto_labels):
        u = np.exp(
            np.array([self.inner_f(x, prototypes[i]) for i in range(len(prototypes)) if x_label != proto_labels[i]])
        )
        inner = np.exp(np.array([self.inner_f(x, p) for p in  prototypes])) 
        den = inner.sum()
        num = u.sum()
        return num/den

    def update_prototypes(self, data, labels, prototypes, proto_labels):
        """
        Performs gradient descent to update the prototypes based on the input data and labels.

        For each data point, the algorithm calculates the distance to each prototype. If the prototype's label matches
        the data point's label, the prototype is pulled towards the data point, otherwise, it is pushed away.

        The update is scaled by a Gaussian kernel based on the distance between the data point and the prototype.

        Args:
            data (np.ndarray): Array of input data points, shape (n_samples, n_features).
            labels (np.ndarray): Array of labels for the input data, shape (n_samples,).
            prototypes (np.ndarray): Array of prototypes, shape (n_prototypes, n_features).
            proto_labels (np.ndarray): Array of labels corresponding to the prototypes, shape (n_prototypes,).

        Returns:
            np.ndarray: Updated prototypes after performing gradient descent.
        """
        for i in range(len(data)):
            xi = data[i]
            x_label = labels[i]
            for j in range(prototypes.shape[0]):
                d = xi - prototypes[j]
                c = 1 / (self.sigma ** 2)
                if proto_labels[j] == x_label:
                    prototypes[j] += self.alpha * (self.Pl(xi, j, prototypes) * self.lst(xi, x_label, prototypes, proto_labels)) * c * d
                else:
                    prototypes[j] -= self.alpha * (self.Pl(xi, j, prototypes) * (1 - self.lst(xi, x_label, prototypes, proto_labels))) * c * d
        return prototypes

    

    def cost(self, data, labels, prototypes, proto_labels):
        """
        Computes the cost function using vectorized operations based on the given data, labels, and prototypes.

        The cost is calculated as the sum of the probabilities that each data point belongs to the incorrect
        prototype (i.e., the prototypes with labels different from the data point's label), normalized by
        the total number of data points.

        Args:
            data (np.ndarray): Array of input data points, shape (n_samples, n_features).
            labels (np.ndarray): Array of labels for the input data, shape (n_samples,).
            prototypes (np.ndarray): Array of prototypes, shape (n_prototypes, n_features).
            proto_labels (np.ndarray): Array of labels corresponding to the prototypes, shape (n_prototypes,).

        Returns:
            float: The computed cost value.
        """
        mismatches = proto_labels != labels
        probabilities = np.array([[self.Pl(xi, j, prototypes) for j in range(len(prototypes))] for xi in data])
        mismatch_probabilities = probabilities[mismatches]
        total_mismatch_probability = np.sum(mismatch_probabilities)
        return total_mismatch_probability / len(data)

    
    def fit(self, train_data, train_labels, decay_scheme = False, show_plot = False):
        self._protolabels, self._prototypes = initialize_prototypes(
            train_data, train_labels, initialization_type=self.initialization_type, num_prototypes_per_class=self.num_prototypes
        )
        self._prototypes = self._prototypes.astype(float)
        loss =[]
 
        for iter in tqdm(range(self.max_iter), desc="Training Progress"):
            if decay_scheme:
                self.alpha = self.alpha*(math.exp(-1*iter/self.max_iter))
            self._prototypes = self.update_prototypes(train_data, train_labels, self._prototypes, self._protolabels)
            predicted = self.predict(train_data)
            val_acc = (np.array(predicted) == train_labels).mean() * 100  
            lr = self.cost(train_data, train_labels, self._prototypes, self._protolabels)

            if iter % 10 == 0:
                logging.info(f'Acc.......{val_acc:.2f}, loss......{lr:.4f}')
            
            loss.append(lr)
        
        logging.info("Training finished")
    
        if show_plot:
            plt.plot(loss)
            plt.ylabel('log likelihood ratio')
            plt.xlabel(' number of iterations')
    

    


