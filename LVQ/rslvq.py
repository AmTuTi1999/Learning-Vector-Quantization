import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from init import initialize_prototypes
from lvq_base import LVQBase
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG, format="%(filename)s:%(lineno)d - %(message)s")


class RSLVQ(LVQBase):
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
        """_summary_

        Args:
            num_prototypes_per_class (_type_): _description_
            initialization_type (str, optional): _description_. Defaults to "mean".
            sigma (int, optional): _description_. Defaults to 1.
            learning_rate (float, optional): _description_. Defaults to 0.05.
            max_iter (int, optional): _description_. Defaults to 100.
        """

        self.max_iter = max_iter
        self.num_prototypes = num_prototypes_per_class
        self.sigma = sigma
        self.initialization_type = initialization_type
        self.alpha = learning_rate
        super(RSLVQ).__init__(
            num_prototypes_per_class,
            initialization_type,
            learning_rate,
            max_iter,
        )

    def update_prototypes(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        prototypes: np.ndarray,
        prototype_labels: np.ndarray,
    ) -> np.ndarray:
        """updating prototypes via gradient ascent

        Args:
            train_data (np.ndarray): train data
            train_labels (np.ndarray): train labels
            prototypes (np.ndarray): class prototypes
            prototype_labels (np.ndarray): prototype labels

        Returns:
            np.ndarray: updated prototypes
        """
        for i, xi in enumerate(train_data):
            x_label = train_labels[i]

            for j, prototype in enumerate(prototypes):
                d = xi - prototype
                c = 1 / (self.sigma * self.sigma)

                if prototype_labels[j] == x_label:
                    prototypes[j] += (
                        self.alpha
                        * (
                            np.subtract(
                                self.correct_classification_probability(
                                    xi, j, x_label, prototypes, prototype_labels
                                ),
                                self.classification_probability(xi, j, prototypes),
                            )
                        )
                        * c
                        * d
                    )
                else:
                    prototypes[j] -= (
                        self.alpha
                        * self.classification_probability(xi, j, prototypes)
                        * c
                        * d
                    )
        return prototypes

    def likelihood_ratio(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        prototypes: np.ndarray,
        protolabels: np.ndarray,
    ) -> float:
        """_summary_

        Args:
            prototypes (_type_): _description_
            protolabels (_type_): _description_
            train_data (_type_): _description_
            train_labels (_type_): _description_

        Returns:
            float: _description_
        """
        numerator = np.sum(
            [
                np.log(
                    np.exp(
                        self.inner_f(xi, prototypes[np.argmax(protolabels == x_label)])
                    )
                )
                for xi, x_label in zip(train_data, train_labels)
            ]
        )
        denominator = np.sum(
            [
                np.log(np.exp(self.inner_f(xi, prototype)))
                for xi in train_data
                for prototype in prototypes
            ]
        )
        return numerator - denominator

    def fit(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        decay_scheme: bool = False,
        show_plot: bool = False,
    ) -> None:
        """Fits the model on training data by iteratively updating prototypes.

        Args:
            train_data (np.ndarray): Training data.
            train_labels (np.ndarray): Training labels.
            show_plot (bool, optional): Whether to display the training plot. Defaults to False.
        """
        logging.info("Initializing prototypes with %s", self.initialization_type)
        prototype_labels, prototypes = initialize_prototypes(
            train_data,
            train_labels,
            initialization_type=self.initialization_type,
            num_prototypes_per_class=self.num_prototypes,
        )
        prototypes = prototypes.astype(float)
        self._prototypes, self._protolabels = prototypes, prototype_labels
        loss = []
        # Using tqdm for progress tracking
        for epoch in tqdm(range(self.max_iter), desc="Training Progress"):
            if decay_scheme:
                self.alpha = self.alpha * (math.exp(-1 * epoch / self.max_iter))
            self._prototypes = self.update_prototypes(
                train_data, train_labels, self._prototypes, self._protolabels
            )
            predicted = self.predict(train_data)
            val_acc = (np.array(predicted) == train_labels).mean() * 100
            lr = self.likelihood_ratio(
                train_data, train_labels, self._prototypes, self._protolabels
            )

            if epoch % 10 == 0:
                logging.info(
                    "Epoch %04d - Accuracy: %.2f%%, Loss: %.4f", epoch, val_acc, lr
                )

            loss.append(lr)

        logging.info("Training finished")

        if show_plot:
            plt.plot(loss)
            plt.ylabel("log likelihood ratio")
            plt.xlabel(" number of iterations")

    def inner_f(self, x: np.ndarray, p: np.ndarray):
        """_summary_

        Args:
            x (np.ndarray): _description_
            p (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        coef = -1 / (2 * (self.sigma * self.sigma))
        dist = (x - p) @ (x - p).T
        return coef * dist

    def inner_derivative(self, x: np.ndarray, p: np.ndarray):
        """_summary_

        Args:
            x (np.ndarray): _description_
            p (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        coef = 1 / (self.sigma * self.sigma)
        diff = x - p
        return coef * diff

    def correct_classification_probability(
        self,
        x: np.ndarray,
        index: int,
        x_label: np.ndarray,
        prototypes: np.ndarray,
        prototype_labels: np.ndarray,
    ) -> float:
        """probability of a point being correctly classified

        Args:
            x (np.ndarray): _description_
            index (int): _description_
            x_label (np.ndarray): _description_
            prototypes (np.ndarray): _description_
            prototype_labels (np.ndarray): _description_

        Returns:
            float: probability of correct classification
        """
        u = np.exp(
            np.array(
                [
                    self.inner_f(x, prototypes[i])
                    for i in range(len(prototypes))
                    if x_label == prototype_labels[i]
                ]
            )
        )
        numerator = np.exp(np.array(self.inner_f(x, prototypes[index])))
        denominator = u.sum()
        return numerator / denominator

    def classification_probability(
        self, x: np.ndarray, index: int, prototypes: np.ndarray
    ) -> float:
        """probability of a point being classified

        Args:
            x (np.ndarray): _description_
            index (int): _description_
            prototypes (np.ndarray): _description_

        Returns:
            float: probability of classification
        """
        inner = np.exp(np.array([self.inner_f(x, p) for p in prototypes]))
        numerator = np.exp(np.array(self.inner_f(x, prototypes[index])))
        denominator = inner.sum()
        return numerator / denominator
