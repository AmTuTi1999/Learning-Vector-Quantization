import numpy as np
import scipy as sc
from abc import abstractmethod
class LVQBase:
    _prototypes: np.ndarray
    _protolabels: np.ndarray
    _weights: np.ndarray | None
    def __init__(
            self, 
            num_prototypes_per_class, 
            initialization_type = 'mean', 
            learning_rate = 0.05,
            max_iter = 100, 
    ):

        self.max_iter = max_iter 
        self.num_prototypes = num_prototypes_per_class
        self.initialization_type = initialization_type
        self.alpha = learning_rate

    @abstractmethod
    def fit(
            self, 
            train_data,
            train_labels
    ) -> None:
        pass

    @property
    def prototypes(self):
        return self._prototypes

    @property
    def protolabels(self):
        return self._protolabels
    
    @abstractmethod
    def update_weights(
        self,
        train_data, 
        train_label, 
        prototypes,  
        proto_labels, 
        weights,
    ) -> np.ndarray:
        pass 

    @abstractmethod
    def cost(
        self, 
        data, 
        labels, 
        prototypes, 
        proto_labels, 
        weights = None,
    ) -> float: 
        pass
          
    @abstractmethod
    def update_prototypes(
        self,
        train_data, 
        train_labels, 
        prototypes, 
        proto_labels,
        weights: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
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

        
    @abstractmethod
    def evaluate(self, test_data, test_labels):
        """predict over test set and outputs test MAE"""
        predicted = np.array(self.predict(test_data))
        val_acc = (predicted == test_labels).mean() * 100 
        return val_acc.item()

    @abstractmethod
    def predict_proba(self, input: np.ndarray) -> list[float]:
        label_prototypes = np.array([self._prototypes[self._protolabels == i] for i in np.unique(self._protolabels)])
        closest_prototypes = np.array([prototypes[np.argmin(np.linalg.norm(input - prototypes, axis=1))] for prototypes in label_prototypes])
        scores = np.linalg.norm(input - closest_prototypes, axis=1)
        return sc.special.softmax(-scores)
 
    @abstractmethod
    def fit(
        self, 
        train_data, 
        train_labels, 
        show_plot = False
    ) -> None:
        pass
    
    