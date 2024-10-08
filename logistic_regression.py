import numpy as np
import warnings

# Logistic Regression Class
class LogisticRegression:
    """
    Logistic Regression implementation using Gradient Descent for binary classification.

    Logistic Regression predicts the probability of a binary class (0 or 1) using a linear combination of 
    input features and the sigmoid activation function.

    Attributes:
    ----------
    learning_rate : float
        The step size for gradient descent. Determines how large the updates to the weights will be.
    iterations : int
        The number of iterations for gradient descent optimization.
    weights : np.ndarray
        Coefficients for the features.
    bias : float
        The intercept term (bias).

    Methods:
    -------
    _sigmoid(x):
        Applies the sigmoid function to convert the linear combination of inputs into a probability.
    
    fit(X, y):
        Trains the logistic regression model by optimizing the weights and bias using gradient descent.
    
    predict(X):
        Predicts binary class labels (0 or 1) for the input data based on the learned model.
    
    Assumptions:
    ------------
    1. Binary classification: This model assumes that the target variable is binary (0 or 1).
    2. Linearity: The relationship between the log-odds of the probability and the input features is assumed to be linear.
    3. No multicollinearity: It is assumed that the independent variables (features) are not highly correlated with each other.
    4. Independence: Observations are assumed to be independent of each other.

    Limitations:
    ------------
    1. Binary classes: Logistic regression works for binary classification. For multi-class problems, it needs to be extended (e.g., using one-vs-rest or softmax for multinomial logistic regression).
    2. Linearity: The model assumes that the log-odds of the probability are a linear function of the input features, which might not hold in real-world data.
    3. Gradient Descent Convergence: The performance of gradient descent depends on the learning rate and might not converge for all datasets.
    4. Feature Scaling: This implementation does not internally handle feature scaling. Feature scaling (standardization) may be necessary to improve convergence, especially for features with different magnitudes.
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initializes the Logistic Regression model with a learning rate and number of iterations for gradient descent.

        Parameters:
        ----------
        learning_rate : float
            The step size for gradient descent updates.
        iterations : int
            The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        """
        Applies the sigmoid activation function to convert linear predictions to probabilities between 0 and 1.

        Parameters:
        ----------
        x : np.ndarray
            The linear combination of inputs (z = X.dot(weights) + bias).

        Returns:
        -------
        np.ndarray
            The sigmoid transformation of input values, giving probabilities between 0 and 1.
        """
        warnings.filterwarnings('ignore')  # suppress sigmoid overflow warning
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clipping for stability

    def fit(self, X, y):
        """
        Trains the logistic regression model by optimizing the weights and bias using gradient descent.

        Parameters:
        ----------
        X : np.ndarray
            The input data (features), with shape (n_samples, n_features).
        y : np.ndarray
            The target values, with shape (n_samples,), containing binary labels (0 or 1).

        Returns:
        -------
        None
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.random.rand(n_features)  # or np.zeros()
        self.bias = 0
        
        # Gradient Descent
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predicts binary class labels (0 or 1) for the input data based on the learned weights and bias.

        Parameters:
        ----------
        X : np.ndarray
            The input data (features), with shape (n_samples, n_features).

        Returns:
        -------
        np.ndarray
            The predicted class labels (0 or 1) based on the 0.5 threshold.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_classes = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_classes)


# Main function for testing
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    
    def eval(y_true, y_predicted):
        """
        Computes evaluation metrics for binary classification: accuracy, precision, recall, and F1-score.

        Parameters:
        ----------
        y_true : np.ndarray
            The actual labels (0 or 1) of the dataset.
        y_predicted : np.ndarray
            The predicted labels (0 or 1) from the logistic regression model.

        Returns:
        -------
        dict
            A dictionary containing accuracy, precision, recall, and F1-score.
        """
        TP = np.sum((y_true == 1) & (y_predicted == 1))
        TN = np.sum((y_true == 0) & (y_predicted == 0))
        FP = np.sum((y_true == 0) & (y_predicted == 1))
        FN = np.sum((y_true == 1) & (y_predicted == 0))
        
        accuracy = (TP + TN) / len(y_true)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        return results

    # Use Breast Cancer dataset from sklearn datasets
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    # Create LogisticRegression object
    regressor = LogisticRegression(learning_rate=0.01, iterations=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    
    # Compute and print the evaluation metrics
    eval_results = eval(y_test, predictions)
    print("Evaluation Metrics:")
    for metric, value in eval_results.items():
        print(f"{metric.capitalize()}: {value:.4f}")