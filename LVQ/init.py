import numpy as np

def initialize_prototypes(
        train_data, 
        train_labels, 
        initialization_type, 
        num_prototypes_per_class, 
        sigma: float = 1.0,
    ):
    """Initialize prototypes based on the provided strategy (mean or random).
    
    Args:
        train_data (np.ndarray): Training data.
        train_labels (np.ndarray): Training labels.
        initialization_type (str): Type of initialization ('mean' or 'random').
        num_prototypes_per_class (int): Number of prototypes to initialize per class.
        sigma (float): Noise scaling factor for prototype perturbation.
    
    Returns:
        np.ndarray: Flattened array of prototype labels.
        np.ndarray: Array of initialized prototypes.
    """
    
    num_features = train_data.shape[1]
    labels = train_labels.astype(int).flatten()
    unique_classes = np.unique(labels)

    total_prototypes = num_prototypes_per_class * len(unique_classes)
    prototypes = []
    proto_labels = []
    
    # Helper function to add random noise
    def add_noise(protos):
        return protos + (0.01 * sigma * np.random.uniform(low=-1.0, high=1.0, size=protos.shape) * protos)
    
    # Mean-based initialization
    if initialization_type == 'mean':
        for class_label in unique_classes:
            class_data = train_data[labels == class_label]
            class_mean = np.mean(class_data, axis=0)
            
            if num_prototypes_per_class == 1:
                # Single prototype: use the mean
                prototypes.append(class_mean)
            else:
                # Multiple prototypes: use the mean and closest points to the mean
                distances = np.linalg.norm(class_data - class_mean, axis=1)
                closest_indices = np.argsort(distances)[1:num_prototypes_per_class]
                closest_points = class_data[closest_indices]
                prototypes.append(np.vstack((class_mean, closest_points)))
            
            proto_labels.extend([class_label] * num_prototypes_per_class)
    
    # Random-based initialization
    elif initialization_type == 'random':
        for class_label in unique_classes:
            class_data = train_data[labels == class_label]
            
            if num_prototypes_per_class == 1:
                # Single random prototype
                random_index = np.random.choice(len(class_data))
                prototypes.append(class_data[random_index])
            else:
                # Multiple random prototypes
                random_indices = np.random.choice(len(class_data), size=num_prototypes_per_class, replace=False)
                prototypes.append(class_data[random_indices])
            
            proto_labels.extend([class_label] * num_prototypes_per_class)
    
    # Convert prototypes and labels into arrays and add noise
    prototypes = np.vstack(prototypes)
    prototypes_noisy = add_noise(prototypes)
    
    return np.array(proto_labels), prototypes_noisy
         

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def weightedL2(a,b,w):
    q = a-b
    return np.sqrt((w*q*q).sum())



def initialize_weights(data):
    weight = np.full(data.shape[1], fill_value = 1/data.shape[1])
    return weight

def sigmoid(x):
    denominator = 1 + np.exp(-x)
    return 1/denominator

def sigmoid_prime(x):
    return sigmoid(x)*(1 - sigmoid(x)) 