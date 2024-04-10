from gp4os.initializers.initializers import rhh
from gp4os.utils.functions import TERMINALS, FUNCTIONS, CONSTANTS
from gp4os.base.population import Population
import pandas as pd
# from gp4os.utils.utils import train_test_split
from gp4os.utils.functions import TERMINALS, FUNCTIONS, CONSTANTS, oversampling_log, Baseline
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, roc_auc_score, recall_score, accuracy_score, precision_score
from imblearn.metrics import geometric_mean_score
from gp4os.algorithm import GP4OS
from gp4os.utils.selection_algorithms import tournament_selection
from gp4os.operators_tree.crossover_operators import crossover_trees
from gp4os.operators_input_set.crossover_operators import crossover_input_set
from gp4os.operators_tree.mutators import mutate_tree_node, mutate_tree_subtree
from gp4os.operators_input_set.mutators import mutate_input_set
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
import os
import csv
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
# from gsmote import GeometricSMOTE
from tqdm import tqdm, trange

def main():

    warnings.filterwarnings("ignore")

    def w_f1_score(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

    def w_recall(y_true, y_pred):
        return  recall_score(y_true, y_pred, average='weighted')

    def w_precision(y_true, y_pred):
         return precision_score(y_true, y_pred, average='weighted')

    def w_gscore(y_true, y_pred):
         return geometric_mean_score(y_true, y_pred, average='weighted')

    metrics = [w_f1_score, w_recall, w_precision, w_gscore, accuracy_score]
    metrics_names = ['f1_score', 'recall', 'precision', 'g_score', 'accuracy']

    estimator = LogisticRegression()


    for i in trange(len(metrics)):

        # ['flare', 'haberman', 'spect', 'ionosphere', 'spectf', 'hungarian', 'diabetes', 'hepatitis',
         # 'appendicitis', 'analcatdata']


        for data_name in tqdm(['flare', 'haberman', 'spect', 'ionosphere', 'spectf', 'hungarian', 'diabetes', 'hepatitis',
         'appendicitis', 'analcatdata_lawsuit']):

            if data_name == 'pima-indians-diabetes':
                data = pd.read_csv('../data/pima-indians-diabetes.txt')
            else:
                data = pd.read_csv(f"../data/{data_name}.tsv",
                             sep='\t')
            X = torch.from_numpy(data.values[:,:-1]).float()
            y = torch.from_numpy(data.values[:,-1]).int()

            for _ in trange(30):

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, stratify=y,
                                                                    random_state=_)

                # oversampling_log(oversampler = SMOTE(), model = estimator,
                #                      X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                #                      name = 'SMOTE', data_file = data_name, iteration=_, path='../log/baseline_bin.csv',
                #                      verbose=1)
                #
                #
                # oversampling_log(oversampler = Baseline(), model = estimator,
                #                      X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                #                      name = 'LR', data_file = data_name, iteration=_, path='../log/baseline_bin.csv',
                #                      verbose=1)
                #
                # oversampling_log(oversampler = BorderlineSMOTE(), model = estimator,
                #                      X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                #                      name = 'BSMOTE', data_file = data_name, iteration=_, path='../log/baseline_bin.csv',
                #                      verbose=1)
                #
                # oversampling_log(oversampler = ADASYN(sampling_strategy='minority'), model = estimator,
                #                  X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                #                  name = 'ADASYN', data_file = data_name, iteration=_, path='../log/baseline_bin.csv',
                #                  verbose=1)
                # try:
                #     oversampling_log(oversampler = SVMSMOTE(), model = estimator,
                #                      X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                #                      name = 'SVMSMOTE', data_file = data_name, iteration=_, path='../log/baseline_bin.csv',
                #                      verbose=1)
                # except:
                #     print('SVMSMOTE: FAILED')
                #
                #
                # oversampling_log(oversampler = GeometricSMOTE(), model = estimator,
                #                  X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                #                  name = 'GSMOTE', data_file = data_name, iteration=_, path='../log/baseline_bin.csv',
                #                  verbose=1)


                X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, stratify=y_test,
                                                                random_state=_)

                y_count = y_train.unique(return_counts=True)
                y_count = (
                y_count[0][y_count[1].argsort(descending=True)], y_count[1][y_count[1].argsort(descending=True)])

                input_set_size = sum([y_count[1][0] - count for count in y_count[1][1:]])

                y_train_extended = torch.concatenate(( y_train,
                    torch.tensor([y_count[0][1] for _ in range( input_set_size)]).float() ))

                pi_eval = {'base_model':estimator,
                      'X_train':X_train,
                      'y_train_extended':y_train_extended,
                      'X_test':X_val,
                      'y_test':y_val,
                      'error_measure':metrics[i]}

                pi_init = {'size': 50,
                           'depth': 6,
                           'FUNCTIONS': FUNCTIONS,
                           'TERMINALS': TERMINALS,
                           'CONSTANTS': CONSTANTS,
                           'p_c': 0.1,
                           'input_set_size': [input_set_size],
                           'umbalanced_obs_ind' : [range(X_train.shape[0])]
                           }
                pi_test = {'base_model':estimator,
                      'X_train':X_train,
                      'y_train_extended':y_train_extended,
                      'X_test':X_test,
                      'y_test':y_test,
                      'error_measure':metrics[i]}

                solver = GP4OS(   pi_eval = pi_eval,
                                  pi_init = pi_init,
                                  pi_test = pi_test,
                                  initializer = rhh,
                                  selector = tournament_selection(pool_size=2),
                                  mutator_tree = mutate_tree_subtree(pi_init['depth'], TERMINALS, CONSTANTS, FUNCTIONS, pi_init['p_c']),
                                  crossover_tree = crossover_trees(FUNCTIONS),
                                  mutator_input_set = mutate_input_set(umbalanced_obs_ind = range(X_train.shape[1])),
                                  crossover_input_set = crossover_input_set,
                                  p_m = 0.2,
                                  p_xo = 0.8,
                                  pop_size = 50,
                                  elitism = True,
                                  seed = _
                      )

                solver.solve( n_iter=50,
                              elitism = True,
                              log = 1,
                              verbose = 1,
                              test_elite = True,
                              log_path = '../log/gm4os_bin.csv',
                              run_info = [f'GP4OS_{metrics_names[i]}', _, data_name],
                              max_depth=17,
                              deep_log = False)




if __name__ == '__main__':
    main()