import time
import random

import torch

from GM4OS.base.population import Population
import numpy as np
from GM4OS.utils.utils_info import logger, verbose_reporter
from GM4OS.base.individual import Individual
from GM4OS.utils.utils_tree import tree_pruning, tree_depth
from sklearn.metrics import classification_report
import pandas as pd



class GM4OS(): #MAXIMIZATION

    """
        Genetic Programming for Optimization and Search (GP4OS) class.

        Attributes
        ----------
        pi_eval : dict
            Dictionary with all the parameters needed for evaluation.

        pi_init : dict
            Dictionary with all the parameters needed for initialization.

        pi_test : dict, optional
            Dictionary with parameters needed for testing (if provided).

        selector : function
            Selector function used for selecting individuals in the population.

        p_m : float, optional
            Probability of mutation. Default is 0.2.

        p_c : float, optional
            Probability of crossover. Default is 0.8.

        elitism : bool, optional
            Flag for elitism. Default is True.

        initializer : function
            Initialization function for creating the initial population.

        mutator_tree : function
            Function for mutating the tree.

        crossover_tree : function
            Function for performing crossover on the tree.

        mutator_input_set : function
            Function for mutating the input set.

        crossover_input_set : function
            Function for performing crossover on the input set.

        pop_size : int, optional
            Size of the population. Default is 100.

        seed : int, optional
            Seed value for randomization. Default is 0.

        Methods
        -------
        solve(n_iter=20, elitism=True, log=0, verbose=0, test_elite=False,
              log_path=None, run_info=None, max_depth=None, max_=True, deep_log=False)
            Solves the optimization problem using genetic programming.
        """

    def __init__(self, pi_eval, pi_init, initializer, selector, mutator_tree, crossover_tree,
                 mutator_input_set, crossover_input_set,
                 p_m=0.2, p_xo=0.8, pop_size=100, elitism=True, seed = 0, pi_test = None):
        #other initial parameters, tipo dataset
        self.pi_eval = pi_eval #dictionary with all the parameters needed for evaluation
        self.pi_init = pi_init  # dictionary with all the parameters needed for evaluation
        self.pi_test = pi_test
        self.selector = selector
        self.p_m = p_m
        self.crossover_tree = crossover_tree
        self.crossover_input_set = crossover_input_set
        self.p_xo = p_xo
        self.elitism = elitism
        self.initializer = initializer
        self.mutator_tree = mutator_tree
        self.mutator_input_set = mutator_input_set
        self.pop_size = pop_size
        self.seed = seed


    def solve(self, n_iter=20, elitism = True, log = 0, verbose = 0,
              test_elite = False, log_path = None, run_info = None,
              max_depth = None, max_ = True, deep_log = False):

        """
                Solves the optimization problem using genetic programming.

                Parameters
                ----------
                n_iter : int, optional
                    Number of iterations. Default is 20.

                elitism : bool, optional
                    Flag for elitism. Default is True.

                log : int, optional
                    Log indicator. Default is 0.

                verbose : int, optional
                    Verbose indicator. Default is 0.

                test_elite : bool, optional
                    Flag for testing the elite individual. Default is False.

                log_path : str, optional
                    Path for logging. Default is None.

                run_info : list or tuple, optional
                    Information related to the run. Default is None.

                max_depth : int, optional
                    Maximum depth for the trees. Default is None.

                max_ : bool, optional
                    Flag for maximization. Default is True.

                deep_log : bool, optional
                    Flag for enabling detailed logging. Default is False.
                """

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        pop = Population(self.initializer(**self.pi_init))
        pop.evaluate(**self.pi_eval)

        end = time.time()

        if max_:
            self.elite = pop.pop[np.argmax(pop.fit)]
        else:
            self.elite = pop.pop[np.argmin(pop.fit)]


        if test_elite and self.pi_test != None:
            self.elite.evaluate_test(**self.pi_test)
        else:
            self.elite.test_fitness = None


        if log != 0:
            if max_:
                logger(log_path, 0, max(pop.fit), end-start, pop.nodes_count,
                    pop_test_report = [self.elite.test_fitness,
                                       self.elite.precision, self.elite.recall, self.elite.gscore, self.elite.f1_score,
                                       self.elite.precision_test, self.elite.recall_test,
                                       self.elite.gscore_test, self.elite.f1_score_test,
                                       self.elite.accuracy_test, self.elite.dist0, self.elite.dist1], run_info = run_info)

            else:
                logger(log_path, 0, min(pop.fit), end - start, pop.nodes_count,
                       pop_test_report=self.elite.test_fitness, run_info=run_info)
        if verbose != 0:
            if max_:
                verbose_reporter(0, max(pop.fit), self.elite.test_fitness, end-start, pop.nodes_count)
            else:
                verbose_reporter(0, min(pop.fit), self.elite.test_fitness, end - start, pop.nodes_count)

        # if deep_log:
        #     report_train = classification_report(self.pi_eval['y_test'], self.elite.predictions, output_dict=True)
        #     report_df_train = pd.DataFrame(report_train).transpose()
        #     report_df_train.to_csv(f'../log/complete_classification_reports'
        #                            f'/{run_info[0]}_{run_info[1]}_{run_info[2]}_train_0.csv')
        #
        #     report_test = classification_report(self.pi_test['y_test'], self.elite.test_pred, output_dict=True)
        #     report_df_test = pd.DataFrame(report_test).transpose()
        #     report_df_test.to_csv(f'../log/complete_classification_reports'
        #                           f'/{run_info[0]}_{run_info[1]}_{run_info[2]}_test_0.csv')


        for it in range(1, n_iter +1, 1):

            offs_pop, start = [], time.time()

            if elitism:

                offs_pop.append(self.elite)

            while len(offs_pop) < pop.size:

                if len(offs_pop) > 4:
                    pass



                if random.random() < self.p_xo:
                    p1, p2 = self.selector(pop), self.selector(pop)
                    while p1 == p2:
                        p1, p2 = self.selector(pop), self.selector(pop)

                    offs1_tree, offs2_tree = self.crossover_tree(p1.tree.repr_, p2.tree.repr_) #two crossovers
                    offs1_input_set, offs2_input_set = self.crossover_input_set(p1.input_choice_repr_, p2.input_choice_repr_)

                    if max_depth != None:

                        if tree_depth(offs1_tree, self.pi_init["FUNCTIONS"]) > max_depth:
                            offs1_tree = tree_pruning(offs1_tree, max_depth, self.pi_init["TERMINALS"],
                                                      self.pi_init["CONSTANTS"], self.pi_init["FUNCTIONS"],
                                                      self.pi_init["p_c"])

                        if tree_depth(offs2_tree, self.pi_init["FUNCTIONS"]) > max_depth:
                            offs2_tree = tree_pruning(offs2_tree, max_depth, self.pi_init["TERMINALS"],
                                                      self.pi_init["CONSTANTS"], self.pi_init["FUNCTIONS"],
                                                      self.pi_init["p_c"])

                        offs_pop.extend([Individual([offs1_tree, offs1_input_set], self.pi_init["FUNCTIONS"],
                                                    self.pi_init["TERMINALS"], self.pi_init["CONSTANTS"]),
                                         Individual([offs2_tree, offs2_input_set], self.pi_init["FUNCTIONS"],
                                                    self.pi_init["TERMINALS"], self.pi_init["CONSTANTS"])])

                else:
                    p1 = self.selector(pop)
                    offs1_tree = self.mutator_tree(p1.tree.repr_)
                    offs1_input_set = self.mutator_input_set(p1.input_choice_repr_)


                    if max_depth != None:

                        if tree_depth(offs1_tree, self.pi_init["FUNCTIONS"]) > max_depth:

                            offs1_tree = tree_pruning(offs1_tree, max_depth, self.pi_init["TERMINALS"],
                                                      self.pi_init["CONSTANTS"], self.pi_init["FUNCTIONS"],
                                                    self.pi_init["p_c"])

                    offs_pop.append(Individual([offs1_tree, offs1_input_set], self.pi_init["FUNCTIONS"],
                                                    self.pi_init["TERMINALS"], self.pi_init["CONSTANTS"]))


            if len(offs_pop) > pop.size:

                offs_pop = offs_pop[:pop.size]

            offs_pop = Population(offs_pop)
            offs_pop.evaluate(**self.pi_eval)

            pop = offs_pop

            end = time.time()

            if max_:
                self.elite = pop.pop[np.argmax(pop.fit)]
            else:
                self.elite = pop.pop[np.argmin(pop.fit)]

            if test_elite and self.pi_test != None:
                self.elite.evaluate_test(**self.pi_test)
            else:
                self.elite.test_fitness = None

            if log != 0:
                if max_:
                    logger(log_path, it, max(pop.fit), end - start, pop.nodes_count,
                           pop_test_report=[self.elite.test_fitness,
                                       self.elite.precision, self.elite.recall, self.elite.gscore, self.elite.f1_score,
                                       self.elite.precision_test, self.elite.recall_test,
                                       self.elite.gscore_test, self.elite.f1_score_test,
                                       self.elite.accuracy_test, self.elite.dist0, self.elite.dist1], run_info=run_info)
                else:
                    logger(log_path, it, min(pop.fit), end - start, pop.nodes_count,
                           pop_test_report=self.elite.test_fitness, run_info=run_info)
            if verbose != 0:
                if max_:
                    verbose_reporter(it, max(pop.fit), self.elite.test_fitness, end - start, pop.nodes_count)
                else:
                    verbose_reporter(it, min(pop.fit), self.elite.test_fitness, end - start, pop.nodes_count)

            # if deep_log:
            #     report_train = classification_report(self.pi_eval['y_test'], self.elite.predictions, output_dict=True)
            #     report_df_train = pd.DataFrame(report_train).transpose()
            #     report_df_train.to_csv(f'../log/complete_classification_reports'
            #                            f'/{run_info[0]}_{run_info[1]}_{run_info[2]}_train_{it}.csv')
            #
            #     report_test = classification_report(self.pi_test['y_test'], self.elite.test_pred, output_dict=True)
            #     report_df_test = pd.DataFrame(report_test).transpose()
            #     report_df_test.to_csv(f'../log/complete_classification_reports'
            #                           f'/{run_info[0]}_{run_info[1]}_{run_info[2]}_test_{it}.csv')
            #
            #
