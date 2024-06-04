import random
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import csv
from sklearn.metrics import recall_score, precision_score, accuracy_score
from imblearn.metrics import geometric_mean_score
import torch

def protected_div(x1, x2):
    """ Implements the division protected against zero denominator

    Performs division between x1 and x2. If x2 is (or has) zero(s), the
    function returns the numerator's value(s).

    Parameters
    ----------
    x1 : torch.Tensor
        The numerator.
    x2 : torch.Tensor
        The denominator.

    Returns
    -------
    torch.Tensor
        Result of protected division between x1 and x2.
    """
    return torch.where(torch.abs(x2) > 0.001, torch.div(x1, x2), torch.tensor(1.0, dtype=x2.dtype, device=x2.device))


def mean_(x1, x2):

    return torch.div(torch.add(x1, x2), 2)

# def w_mean_(x1, x2):
#
#     r = random.random()
#
#     return torch.add(torch.mul(x1, r), torch.mul(x2, r))


FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity': 2},
    'mean': {'function': lambda x, y: mean_(x, y), 'arity': 2},
    'tan': {'function': lambda x: torch.tan(x), 'arity': 1},
    'sin': {'function': lambda x: torch.sin(x), 'arity': 1},
    'cos': {'function': lambda x: torch.cos(x), 'arity': 1},
}

TERMINALS = {
    'input_1': lambda inputs: inputs[:,0,:],
    'input_2': lambda inputs: inputs[:,1,:],
}

CONSTANTS = {
    'constant_2': lambda inputs: torch.tensor(2).float(),
    'constant_3': lambda inputs: torch.tensor(3).float(),
    'constant_4': lambda inputs: torch.tensor(4).float(),
    'constant_5': lambda inputs: torch.tensor(5).float(),
    'constant__1': lambda inputs: torch.tensor(-1).float(),
    'constant_1': lambda inputs: torch.tensor(1).float()
}


def w_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def oversampling_log(oversampler, model, X_train, y_train, X_test, y_test,
                     name, data_file, iteration, path = '../log/baseline_mc.csv', verbose = 0):

    resampled_X, resampled_y = oversampler.fit_resample(X_train, y_train)
    model.fit(resampled_X, resampled_y)
    pred_rs = model.predict(X_test)
    if verbose > 0:
        print(f'BASELINE SCORE {name}:', w_f1_score(y_test, pred_rs))


    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, iteration, data_file, w_f1_score(y_test, pred_rs),
                         recall_score(y_test, pred_rs, average='weighted'),
                         precision_score(y_test, pred_rs, average='weighted'),
                        geometric_mean_score(y_test, pred_rs, average='weighted'),
                         accuracy_score(y_test, pred_rs)])


class Baseline():

    def __init__(self):
        pass

    def fit_resample(self, X, y):

        return X, y


dist = torch.nn.PairwiseDistance()

