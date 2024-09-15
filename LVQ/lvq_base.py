"""GLVQ Models"""

import logging
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

logging.basicConfig(level=logging.DEBUG, format="%(filename)s:%(lineno)d - %(message)s")


class LVQBase:
    """_summary_

    Returns:
        _type_: _description_
    """

    _prototypes: np.ndarray
    _protolabels: np.ndarray
    _weights: np.ndarray | None

    def __init__(
        self,
        num_prototypes_per_class,
        initialization_type: str = "mean",
        learning_rate: float = 0.05,
        max_iter: int = 100,
    ):
        """_summary_

        Args:
            num_prototypes_per_class (_type_): _description_
            initialization_type (str, optional): _description_. Defaults to "mean".
            learning_rate (float, optional): _description_. Defaults to 0.05.
            max_iter (int, optional): _description_. Defaults to 100.
        """
        self.max_iter = max_iter
        self.num_prototypes = num_prototypes_per_class
        self.initialization_type = initialization_type
        self.alpha = learning_rate

    @property
    def prototypes(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        return self._prototypes

    @property
    def protolabels(self) -> np.ndarray:
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._protolabels

    def predict(self, data: np.ndarray) -> list | np.ndarray:
        """_summary_

        Args:
            data (np.ndarray): _description_

        Returns:
            list | np.ndarray: _description_
        """
        assert len(data.shape) == 2
        distances = np.linalg.norm(data[:, None] - self._prototypes, axis=2)
        return self._protolabels[np.argmin(distances, axis=1)].tolist()

    def evaluate(self, test_data, test_labels):
        """predict over test set and outputs test MAE"""
        predicted = np.array(self.predict(test_data))
        val_acc = (predicted == test_labels).mean() * 100
        return val_acc.item()

    def predict_proba(self, x: np.ndarray) -> list[float]:
        """_summary_

        Args:
            input (np.ndarray): _description_

        Returns:
            list[float]: _description_
        """
        label_prototypes = np.array(
            [
                self._prototypes[self._protolabels == i]
                for i in np.unique(self._protolabels)
            ]
        )
        closest_prototypes = []
        for prototypes in label_prototypes:
            distances = np.linalg.norm(x[:, None] - prototypes, axis=2)
            closest_idxs = np.argmin(distances, axis=1)
            closest_prototypes.append(prototypes[closest_idxs])
        closest_prototypes = np.array(closest_prototypes).transpose(
            1, 0, 2
        )  # (n_samples, n_labels, n_features)
        scores = np.linalg.norm(x[:, None] - closest_prototypes, axis=2)

        return sc.special.softmax(-scores, axis=1)

    @abstractmethod
    def fit(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        decay_scheme: bool = False,
        show_plot: bool = False,
    ) -> None:
        """_summary_

        Args:
            train_data (_type_): _description_
            train_labels (_type_): _description_
            show_plot (bool, optional): _description_. Defaults to False.
        """

    def cross_validate(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        n_folds: int,
        show_plot: bool = False,
    ):
        """
        Perform k-fold cross-validation to assess model stability.

        This method splits the dataset into k folds, trains the model on k-1 folds,
        and tests it on the remaining fold. It repeats this process for each fold and
        calculates accuracy for each iteration. It also calculates and logs the mean
        and variance of the accuracies.

        Parameters:
        -----------
        data : np.ndarray
            The dataset to train on, with shape (n_samples, n_features).
        labels : np.ndarray
            The true labels for the data, with shape (n_samples,).
        n_folds : int
            Number of folds to use for cross-validation.
        show_plot : bool, optional
            If True, a plot will display the legend for each fold (default is False).

        Returns:
        --------
        None
        """
        fold_accuracies = []
        fold_train_indices = []
        fold_test_indices = []
        indices = np.arange(len(data))
        logging.info("Starting %d-fold cross-validation", n_folds)

        for fold in range(n_folds):
            np.random.shuffle(indices)
            split_point = int(len(indices) / n_folds)
            train_idx, test_idx = indices[split_point:], indices[:split_point]
            fold_train_indices.append(train_idx)
            fold_test_indices.append(test_idx)
            train_data, test_data = data[train_idx], data[test_idx]
            train_labels_fold, test_labels_fold = labels[train_idx], labels[test_idx]
            self.fit(train_data, train_labels_fold)

            if show_plot:
                plt.legend([f"Fold {fold+1}"])

            predictions = self.predict(test_data)

            if len(predictions) == 0:
                logging.warning(
                    "No predictions made for fold %d, stopping early.", fold + 1
                )
                break

            accuracy = (predictions == test_labels_fold).mean().item()
            fold_accuracies.append(accuracy)
            logging.info("Fold %d: Accuracy = %.4f", fold + 1, accuracy)

        mean_accuracy = np.mean(fold_accuracies)
        accuracy_variance = np.var(fold_accuracies)
        logging.info("Accuracies: %s", str(fold_accuracies))
        logging.info("Mean Accuracy: %s", str(mean_accuracy))
        logging.info("Accuracy Variance: %s", str(accuracy_variance))

        logging.info("Finished Cross Validation")

        if show_plot:
            plt.show()
