from gp4os.initializers.initializers import rhh
from gp4os.utils.functions import TERMINALS, FUNCTIONS, CONSTANTS
from gp4os.base.population import Population
import pandas as pd
# from gp4os.utils.utils import train_test_split
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from gp4os.algorithm import GP4OS
from gp4os.utils.selection_algorithms import tournament_selection
from gp4os.operators_tree.crossover_operators import crossover_trees
from gp4os.operators_input_set.crossover_operators import crossover_input_set
from gp4os.operators_tree.mutators import mutate_tree_node
from gp4os.operators_input_set.mutators import mutate_input_set
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
import os
import csv
from tqdm import tqdm, trange

def main():

    warnings.filterwarnings("ignore")

    # def macro_f1_score(y_true, y_pred):
    #
    #     return f1_score(y_true, y_pred, average='macro')

    # 'pima-indians-diabetes', 'flare',  'haberman',

    # 'spect', 'ionosphere
    # spectf
    # 'pima-indians-diabetes', 'flare', 'haberman', 'spect', 'ionosphere', 'spectf'
    # 'flare', 'haberman'

    for data_name in tqdm(['flare', 'haberman', 'spect', 'ionosphere', 'spectf', 'hungarian', 'diabetes', 'appendicitis', 'analcatdata_lawsuit', 'hepatitis']):

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
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, stratify=y_test,
                                                            random_state=_)


            y_count = y_train.unique(return_counts=True)

            input_set_size = abs(y_count[1][0] - y_count[1][1])

            def min_class_f1_score(y_true, y_pred):
                return f1_score(y_true, y_pred, average='binary', pos_label=int(y_count[0][1]))
                # report = classification_report(y_true, y_pred, output_dict=True)
                # return report[str(int(y_count[0][1]))]['f1-score']

            # model = clone(LogisticRegression(random_state=_))
            # model.fit(X_train, y_train)
            # pred = model.predict(X_test)

            # with open('../log/baseline_complete.csv', 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(['LR', _, data_name, min_class_f1_score(y_test, pred)])

            # report = classification_report(y_test, pred, output_dict=True)
            # report_df = pd.DataFrame(report).transpose()
            # report_df.to_csv(f'../log/classification_reports_3/LR_{_}_{data_name}.csv')
            # #
            # oversampler = clone(SMOTE(random_state=_))
            # resampled_X, resampled_y = oversampler.fit_resample(X_train, y_train)
            # model = clone(LogisticRegression(random_state=_))
            # model.fit(resampled_X, resampled_y)
            # pred_rs = model.predict(X_test)

            # with open('../log/baseline_complete.csv', 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(['LR_SMOTE', _, data_name, min_class_f1_score(y_test, pred_rs)])

            # report = classification_report(y_test, pred_rs, output_dict=True)
            # report_df = pd.DataFrame(report).transpose()
            # report_df.to_csv(f'../log/classification_reports_3/LR_SMOTE_{_}_{data_name}.csv')




            y_train_extended = torch.concatenate(( y_train,
                torch.tensor([y_count[0][1] for _ in range( input_set_size)]).float() ))

            pi_eval = {'base_model':LogisticRegression(),
                  'X_train':X_train,
                  'y_train_extended':y_train_extended,
                  'X_test':X_val,
                  'y_test':y_val,
                  'error_measure':min_class_f1_score}

            pi_init = {'size': 50,
                       'depth': 4,
                       'FUNCTIONS': FUNCTIONS,
                       'TERMINALS': TERMINALS,
                       'CONSTANTS': CONSTANTS,
                       'p_c': 0.1,
                       'input_set_size': [input_set_size],
                       'umbalanced_obs_ind': [((y_train == y_count[0][1]).nonzero(as_tuple = True))[0].tolist()]
                       }

            pi_test = {'base_model':LogisticRegression(),
                  'X_train':X_train,
                  'y_train_extended':y_train_extended,
                  'X_test':X_test,
                  'y_test':y_test,
                  'error_measure':min_class_f1_score}

            solver = GP4OS(   pi_eval = pi_eval,
                              pi_init = pi_init,
                              pi_test = pi_test,
                              initializer = rhh,
                              selector = tournament_selection(pool_size=2),
                              mutator_tree=mutate_tree_node(4, TERMINALS, CONSTANTS, FUNCTIONS, 0.3),
                              crossover_tree=crossover_trees(FUNCTIONS),
                              mutator_input_set=mutate_input_set(umbalanced_obs_ind=((y_train == y_count[0][1]).nonzero(as_tuple = True))[0].tolist(), p_m=0.2),
                              crossover_input_set=crossover_input_set,
                              p_m = 0.2,
                              p_xo = 0.8,
                              pop_size = 50,
                              elitism = True,
                              seed = _
                  )

            solver.solve( n_iter=50,
                          elitism = True,
                          log = 1,
                          verbose = 0,
                          test_elite = True,
                          log_path =  '../log/gp4os_FIXED.csv',
                          run_info = ['GP4OS_FIXED', _, data_name],
                          max_depth=8,
                          deep_log = True)

            # report = classification_report(y_test, solver.elite.test_pred, output_dict=True)
            # report_df = pd.DataFrame(report).transpose()
            # report_df.to_csv(f'../log/classification_reports'
            #                  f'/GP4OS_{_}_{data_name}_minclass.csv')



if __name__ == '__main__':
    main()