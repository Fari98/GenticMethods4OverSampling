import torch
import math

def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0):
    """ Splits X and y tensors into train and test subsets

    This method replicates the behaviour of Sklearn's 'train_test_split'.

    Parameters
    ----------
    X : torch.Tensor
        Input data instances,
    y : torch.Tensor
        Target vector.
    p_test : float (default=0.3)
        The proportion of the dataset to include in the test split.
    shuffle : bool (default=True)
        Whether or not to shuffle the data before splitting.
    indices_only : bool (default=False)
        Whether or not to return only the indices representing training and test partition.
    seed : int (default=0)
        The seed for random numbers generators.

    Returns
    -------
    X_train : torch.Tensor
        Training data instances.
    y_train : torch.Tensor
        Training target vector.
    X_test : torch.Tensor
        Test data instances.
    y_test : torch.Tensor
        Test target vector.
    train_indices : torch.Tensor
        Indices representing the training partition.
    test_indices : torch.Tensor
    Indices representing the test partition.
    """
    # Sets the seed before generating partition's indexes
    torch.manual_seed(seed)
    # Generates random indices
    if shuffle:
        indices = torch.randperm(X.shape[0])
    else:
        indices = torch.arange(0, X.shape[0], 1)
    # Splits indices
    split = int(math.floor(p_test * X.shape[0]))
    train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        # Generates train/test partitions
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return


def flatten(data):
    """
        Flattens a nested tuple structure.

        Parameters
        ----------
        data : tuple
            Input nested tuple data structure.

        Yields
        ------
        object
            Flattened data element by element.
    """

    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data


def tangent_distance_weight_based(y_true, y_pred):
    """
        Calculates the tangent distance based on predicted and true values.

        Parameters
        ----------
        y_true : torch.Tensor
            True values.
        y_pred : torch.Tensor
            Predicted values.

        Returns
        -------
        float
            Tangent distance weighted calculation.
    """

    y_count = y_true.unique(return_counts=True)

    idxs_0 = (y_true == y_count[0][0]).nonzero()
    idxs_1 = (y_true == y_count[0][1]).nonzero()

    dist_0 = torch.eq(y_true[idxs_0], y_pred[idxs_0])
    dist_1 = torch.eq(y_true[idxs_1], y_pred[idxs_1])

    return float(torch.sum(torch.tan(dist_0) / (2 * len(idxs_0))) + torch.sum(torch.tan(dist_1) / (2 * len(idxs_1))))


def bound_value(vector, min_val, max_val):
    """
        Constrains the values within a specific range.

        Parameters
        ----------
        vector : torch.Tensor
            Input tensor to be bounded.
        min_val : float
            Minimum value for bounding.
        max_val : float
            Maximum value for bounding.

        Returns
        -------
        torch.Tensor
            Tensor with values bounded between min_val and max_val.
    """

    return torch.clamp(vector, min_val, max_val)

