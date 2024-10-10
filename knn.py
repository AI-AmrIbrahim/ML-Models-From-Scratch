# Imports
import numpy as np
from collections import Counter

# Define Euclidean Distance
def euclidean_distance(x1, x2):
        """
        Computes the Euclidean distance between two data points.

        Parameters:
        ----------
        x1 : np.ndarray
            The first data point.
        x2 : np.ndarray
            The second data point.

        Returns:
        -------
        float
            The Euclidean distance between the two points.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

# K Nearest Neighbors Class
class KNN:
    """
    K-Nearest Neighbors (KNN) algorithm for classification.

    KNN is a non-parametric algorithm that predicts the label of a data point based on the majority label
    of its k-nearest neighbors in the feature space.

    Attributes:
    ----------
    k : int
        The number of neighbors to consider for making the prediction.
    X_train : np.ndarray
        The training data (features).
    y_train : np.ndarray
        The training labels.

    Methods:
    -------
    _euclidean_distance(x1, x2):
        Computes the Euclidean distance between two data points.
    
    fit(X, y):
        Stores the training data and labels.
    
    predict(X):
        Predicts the class labels for the given input data based on the k-nearest neighbors.
    
    predict_single(x):
        Predicts the class label for a single data point using majority voting among the k-nearest neighbors.

    Assumptions:
    ------------
    1. The algorithm assumes that the input data is numerical and well-scaled.
    2. KNN works best when the data is low-dimensional, as high-dimensional data can lead to poor performance due to the "curse of dimensionality."
    3. The performance of the algorithm depends on a well-chosen `k` value and a proper distance metric.
    
    Limitations:
    ------------
    1. Computational Efficiency: KNN is computationally expensive, especially for large datasets, since it requires calculating the distance between the test instance and all training instances for every prediction.
    2. Curse of Dimensionality: KNN performs poorly in high-dimensional spaces because the distance between points becomes less informative as the number of dimensions increases.
    3. Feature Scaling: The current implementation does not handle feature scaling. Without normalization or standardization, features with larger ranges may dominate the distance metric.
    4. Imbalanced Data: KNN may struggle with imbalanced datasets, as the majority class may dominate the prediction based on sheer quantity, even if the minority class is more relevant.
    5. Fixed K: The choice of `k` can greatly affect the model's performance. This implementation requires manually tuning the parameter, which may not be optimal for all datasets.
    6. Distance Metric Limitation: The current implementation only supports the Euclidean distance metric. Other distance metrics (e.g., Manhattan, Minkowski) might work better depending on the dataset.
    """

    def __init__(self, k = 3):
        self.k = k
    
    def fit(self, X, y):
        """
        Stores the training data and labels for use during prediction.

        Parameters:
        ----------
        X : np.ndarray
            The training data (features), with shape (n_samples, n_features).
        y : np.ndarray
            The training labels, with shape (n_samples,).
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """
        Predicts the class labels for the input data using the k-nearest neighbors algorithm.

        Parameters:
        ----------
        X : np.ndarray
            The input data (features), with shape (n_samples, n_features).

        Returns:
        -------
        np.ndarray
            The predicted class labels.
        """
        predictions = [self._singular_predict(x) for x in X]
        return np.array(predictions)
    
    def _singular_predict(self, x):
        """
        Predicts the class label for a single data point by finding its k-nearest neighbors.

        Parameters:
        ----------
        x : np.ndarray
            The input data point, with shape (n_features,).

        Returns:
        -------
        int
            The predicted class label.
        """
        # Compute distances to all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote: Return the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Main function for testing
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    
    # Create evaluation function for accuracy
    def accuracy(y_true, y_pred):
        """
        Computes the classification accuracy.

        Parameters:
        ----------
        y_true : np.ndarray
            The actual labels of the dataset.
        y_pred : np.ndarray
            The predicted labels from the KNN model.

        Returns:
        -------
        float
            The accuracy score (the percentage of correct predictions).
        """
        return np.sum(y_true == y_pred) / len(y_true)
    
    # Use Breast Cancer dataset from sklearn datasets
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    # Create and fit the KNN classifier
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    
    # Predict on the test set
    predictions = knn.predict(X_test)
    
    # Print the accuracy
    print(f"Accuracy: {accuracy(y_test, predictions):.4f}")