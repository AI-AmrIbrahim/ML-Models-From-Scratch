#%% Imports
import numpy as np
#%% Linear Regression Class
class LinearRegression:
    #%% init method
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    #%% fit method
    def fit(self, X, y):
        # Number of samples and features
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features) # zeros or random.rand
        self.bias = 0
        
        # Gradient Descent