from gp4os.initializers.initializers import rhh
from gp4os.utils.functions import TERMINALS, FUNCTIONS, CONSTANTS
from gp4os.base.population import Population
import pandas as pd
# from gp4os.utils.utils import train_test_split
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from gp4os.algorithm import GP4OS
from gp4os.utils.selection_algorithms import tournament_selection, tournament_selection_min
from gp4os.operators_tree.crossover_operators import crossover_trees
from gp4os.operators_input_set.crossover_operators import crossover_input_set
from gp4os.operators_tree.mutators import mutate_tree_node
from gp4os.operators_input_set.mutators import mutate_input_set, multiclass_mutate_input_set
from imblearn.over_sampling import SMOTE
from sklearn.base import clone


def main():

    for _ in range(7, 12):



        warnings.filterwarnings("ignore")

        data = pd.read_csv(f"../data/analcatdata_dmft.tsv",
                           sep='\t')

        # data = pd.read_csv('../data/pima-indians-diabetes.txt')
        X = torch.from_numpy(data.values[:,:-1]).float()
        y = torch.from_numpy(data.values[:,-1]).int()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, stratify=y, random_state=_)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, stratify=y_test, random_state=_)


        y_count = y_train.unique(return_counts=True)
        y_count = (y_count[0][y_count[1].argsort(descending = True)], y_count[1][y_count[1].argsort(descending = True)])


        input_set_size = sum([y_count[1][0] - count for count in y_count[1][1:]])

        def w_f1_score(y_true, y_pred):
            return f1_score(y_true, y_pred, average = 'weighted')


        model = clone(LogisticRegression(random_state=_))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print('BASELINE SCORE:', w_f1_score(y_test, pred))

        oversampler = SMOTE(random_state=_)
        resampled_X, resampled_y = oversampler.fit_resample(X_train, y_train)
        model = clone(LogisticRegression(random_state=_))
        model.fit(resampled_X, resampled_y)
        pred = model.predict(X_test)
        print('BASELINE SCORE SMOTE:', w_f1_score(y_test, pred))


        # pop_repr_  = rhh(100, 4,  FUNCTIONS, TERMINALS, input_set_size, range(X_train.shape[0]))
        # pop = Population(pop_repr_)


        y_train_extended = torch.concatenate(( y_train,
            torch.tensor([y_count[0][1] for _ in range(input_set_size)]).float() ))

        # pop.evaluate(LogisticRegression(), X_train, y_train_extended, X_test, y_test, f1_score)

        pi_eval = {'base_model':LogisticRegression(),
              'X_train':X_train,
              'y_train_extended':y_train_extended,
              'X_test':X_val,
              'y_test':y_val,
              'error_measure':w_f1_score}

        pi_init = {'size': 50,
                   'depth': 4,
                   'FUNCTIONS':FUNCTIONS,
                   'TERMINALS':TERMINALS,
                   'CONSTANTS' : CONSTANTS,
                   'p_c' : 0.1,
                   'input_set_size' : [input_set_size],
                   'umbalanced_obs_ind' : [(y_train == y_count[0][i]).nonzero().squeeze().tolist() for i in range(1, len(y_count[1]))] # no as_tuple .squeeze()
                   }

        pi_test = {'base_model':LogisticRegression(),
              'X_train':X_train,
              'y_train_extended':y_train_extended,
              'X_test':X_test,
              'y_test':y_test,
              'error_measure':w_f1_score}

        # ordered_indices = [0]
        ordered_indices = [y_count[1][0] - count for count in y_count[1][1:]]


        solver = GP4OS(   pi_eval = pi_eval,
                          pi_init = pi_init,
                          pi_test = pi_test,
                          initializer = rhh,
                          selector = tournament_selection_min(pool_size=2),
                          mutator_tree = mutate_tree_node(4, TERMINALS, CONSTANTS, FUNCTIONS, 0.3),
                          crossover_tree = crossover_trees(FUNCTIONS),
                          mutator_input_set = multiclass_mutate_input_set(indices_minority_classes = list(zip(ordered_indices, [(y_train == y).nonzero().squeeze().tolist() for y in y_count[0]])) , p_m = 0.5),
                          # mutator_input_set = mutate_input_set(umbalanced_obs_ind = (y_train == y_count[0][1]).nonzero().squeeze().tolist(), p_m = 0.2),
                          crossover_input_set = crossover_input_set,
                          p_m = 0.2,
                          p_xo = 0.8,
                          pop_size = 50,
                          elitism = True,
                          seed = _
              )

        solver.solve( n_iter=15,
                      elitism = True,
                      log = 1,
                      verbose = 1,
                      test_elite = True,
                      log_path = '../log/test.csv',
                      run_info = ['run', 'dataset_name', 0],
                      max_depth = 8,
                      max_ = True)

        solver.elite.tree.print_tree_representation()


if __name__ == '__main__':
    main()