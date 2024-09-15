"""Soft Nearest Prototype Classifier"""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from init import initialize_prototypes
from lvq_base import LVQBase
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG, format="%(filename)s:%(lineno)d - %(message)s")


class SNPC(LVQBase):
    """_summary_

    Args:
        LVQBase (_type_): _description_
    """

    def __init__(
        self,
        num_prototypes_per_class: int,
        initialization_type: str = "mean",
        sigma: float = 1.0,
        learning_rate: float = 0.05,
        max_iter: int = 100,
    ):

        self.max_iter = max_iter
        self.num_prototypes = num_prototypes_per_class
        self.sigma = sigma
        self.initialization_type = initialization_type
        self.alpha = learning_rate
        super(SNPC).__init__(
            num_prototypes_per_class,
            initialization_type,
            learning_rate,
            max_iter,
        )

    def inner_f(self, x: np.ndarray, p: np.ndarray):
        """_summary_

        Args:
            x (_type_): _description_
            p (_type_): _description_

        Returns:
            _type_: _description_
        """
        coef = -1 / (2 * (self.sigma * self.sigma))
        dist = (x - p) @ (x - p).T
        return coef * dist

    def inner_derivative(self, x: np.ndarray, p: np.ndarray):
        """_summary_

        Args:
            x (_type_): _description_
            p (_type_): _description_

        Returns:
            _type_: _description_
        """
        coef = 1 / (self.sigma * self.sigma)
        diff = x - p
        return coef * diff

    def classification_probability(
        self, x: np.ndarray, index: int, prototypes: np.ndarray
    ):
        """_summary_

        Args:
            x (_type_): _description_
            index (_type_): _description_
            prototypes (_type_): _description_

        Returns:
            _type_: _description_
        """
        inner = np.exp(np.array([self.inner_f(x, p) for p in prototypes]))
        numerator = np.exp(np.array(self.inner_f(x, prototypes[index])))
        denominator = inner.sum()
        return numerator / (denominator)

    def lst(
        self,
        x: np.ndarray,
        x_label: np.ndarray,
        prototypes: np.ndarray,
        proto_labels: np.ndarray,
    ):
        """_summary_

        Args:
            x (_type_): _description_
            x_label (_type_): _description_
            prototypes (_type_): _description_
            proto_labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        u = np.exp(
            np.array(
                [
                    self.inner_f(x, prototypes[i])
                    for i in range(len(prototypes))
                    if x_label != proto_labels[i]
                ]
            )
        )
        inner = np.exp(np.array([self.inner_f(x, p) for p in prototypes]))
        den = inner.sum()
        num = u.sum()
        return num / den

    def update_prototypes(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        prototypes: np.ndarray,
        proto_labels: np.ndarray,
    ):
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
        for i, xi in enumerate(train_data):
            x_label = train_labels[i]
            for j, prototype in enumerate(prototypes):
                d = xi - prototype
                c = 1 / (self.sigma**2)
                prob = self.classification_probability(xi, j, prototypes)
                lst_value = self.lst(xi, x_label, prototypes, proto_labels)

                if proto_labels[j] == x_label:
                    prototypes[j] += self.alpha * prob * lst_value * c * d
                else:
                    prototypes[j] -= self.alpha * prob * (1 - lst_value) * c * d
        return prototypes

    def cost(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        prototypes: np.ndarray,
        proto_labels: np.ndarray,
    ):
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
        mismatches = proto_labels != train_labels
        probabilities = np.array(
            [
                [
                    self.classification_probability(xi, j, prototypes)
                    for j in range(len(prototypes))
                ]
                for xi in train_data
            ]
        )
        mismatch_probabilities = probabilities[mismatches]
        total_mismatch_probability = np.sum(mismatch_probabilities)
        return total_mismatch_probability / len(train_data)

    def fit(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        decay_scheme: bool = False,
        show_plot: bool = False,
    ):
        """_summary_

        Args:
            train_data (np.ndarray): _description_
            train_labels (np.ndarray): _description_
            decay_scheme (bool, optional): _description_. Defaults to False.
            show_plot (bool, optional): _description_. Defaults to False.
        """
        self._protolabels, self._prototypes = initialize_prototypes(
            train_data,
            train_labels,
            initialization_type=self.initialization_type,
            num_prototypes_per_class=self.num_prototypes,
        )
        self._prototypes = self._prototypes.astype(float)
        loss = []

        for epoch in tqdm(range(self.max_iter), desc="Training Progress"):
            if decay_scheme:
                self.alpha = self.alpha * (math.exp(-1 * epoch / self.max_iter))
            self._prototypes = self.update_prototypes(
                train_data, train_labels, self._prototypes, self._protolabels
            )
            predicted = self.predict(train_data)
            val_acc = (np.array(predicted) == train_labels).mean() * 100
            lr = self.cost(
                train_data, train_labels, self._prototypes, self._protolabels
            )

            if epoch % 10 == 0:
                logging.info("Acc.......%.2f, loss......%.4f", val_acc, lr)

            loss.append(lr)

        logging.info("Training finished")

        if show_plot:
            plt.plot(loss)
            plt.ylabel("log likelihood ratio")
            plt.xlabel(" number of iterations")
