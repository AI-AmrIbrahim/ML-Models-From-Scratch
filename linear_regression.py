# Imports
import numpy as np
# Linear Regression Class
class LinearRegression:
    """
    A simple implementation of Linear Regression using Gradient Descent.

    Linear Regression aims to model the relationship between a dependent variable (y) and 
    one or more independent variables (X) by fitting a linear equation to the observed data.

    Attributes:
    ----------
    learning_rate : float
        The step size for gradient descent. Determines how large the updates to the weights will be.
    iterations : int
        Number of iterations for gradient descent optimization.
    weights : np.ndarray
        Coefficients (slopes) for the features.
    bias : float
        The intercept term (bias).
    
    Methods:
    -------
    fit(X, y):
        Trains the linear regression model by optimizing the weights and bias using gradient descent.
    predict(X):
        Predicts the target values for the given input data X using the learned weights and bias.
    
    Assumptions:
    ------------
    1. Linearity: The relationship between the features (X) and the target (y) is assumed to be linear.
    2. Independence: Observations are assumed to be independent of each other.
    3. Homoscedasticity: The variance of the residuals (errors) is assumed to be constant across all values of the independent variables.
    4. No Multicollinearity: Independent variables are not highly correlated with each other.
    5. Normality of Errors: The residuals (differences between observed and predicted values) are normally distributed.
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    # fit method
    def fit(self, X, y):
        """
        Trains the Linear Regression model using Gradient Descent.

        Parameters:
        ----------
        X : np.ndarray
            The input data (features), with shape (n_samples, n_features).
        y : np.ndarray
            The target values, with shape (n_samples,).

        Returns:
        -------
        None
        """
        # Number of samples and features
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.random.rand(n_features) # or np.zeros()
        self.bias = 0
        
        # Gradient Descent
        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    # predict method
    def predict(self, X):
        """
        Predicts the target values for the given input data X.

        Parameters:
        ----------
        X : np.ndarray
            The input data (features), with shape (n_samples, n_features).

        Returns:
        -------
        np.ndarray
            The predicted target values, with shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias


# Main function for testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    
    # Create MSE function
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    # Create R2 function
    def r2_score(y_true, y_pred):
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0, 1]
        return corr ** 2

    
    # Generate synthetic data
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Create a LinearRegression object
    regressor = LinearRegression(learning_rate=0.01, iterations=1000)
    # Fit the model
    regressor.fit(X_train, y_train)
    # Predict on the training data
    predictions = regressor.predict(X_test)

    # Plot results
    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()

    # Print Weights, Bias, MSE, and R2
    print(f"Weights: {regressor.weights}")
    print(f"Bias: {regressor.bias}")
    print(f"MSE: {mean_squared_error(y_test, predictions)}")
    print(f"R^2: {r2_score(y_test, predictions)}")