import torch
from gp4os.base.tree import Tree
from gp4os.base.input_set import Input_Set
# from sklearn.base import clone
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from imblearn.metrics import geometric_mean_score

class Individual():
    """
        Class representing an individual in a genetic programming population.

        Methods
        -------
        __init__(individual_repr_, FUNCTIONS, TERMINALS, CONSTANTS)
            Initializes an Individual object.

        evaluate(base_model, X_train, y_train_extended, X_test, y_test, error_measure)
            Evaluates the individual's fitness using a base model and train/test datasets.

        evaluate_test(base_model, X_train, y_train_extended, X_test, y_test, error_measure)
            Evaluates the individual's test fitness using a base model and train/test datasets.
        """

    def __init__(self, individual_repr_,  FUNCTIONS, TERMINALS, CONSTANTS):
        """
               Initializes an Individual object.

               Parameters
               ----------
               individual_repr_ : tuple
                   Representation of the individual consisting of tree and input choice.

               FUNCTIONS : dict
                   Dictionary of functions allowed in the tree.

               TERMINALS : dict
                   Dictionary of terminal symbols allowed in the tree.

               CONSTANTS : dict
                   Dictionary of constant values allowed in the tree.
               """

        self.tree_repr_ = individual_repr_[0]
        self.input_choice_repr_ = individual_repr_[1]

        self.tree = Tree(individual_repr_[0], FUNCTIONS, TERMINALS, CONSTANTS)
        self.input_choice = Input_Set(individual_repr_[1])



    def evaluate(self, base_model, X_train, y_train_extended, X_test, y_test, error_measure):
        """
                Evaluates the individual's fitness using a base model and train/test datasets.

                Parameters
                ----------
                base_model : object
                    Base model for evaluating the individual.

                X_train : tensor
                    Training input data.

                y_train_extended : tensor
                    Extended training target data.

                X_test : tensor
                    Test input data.

                y_test : tensor
                    Test target data.

                error_measure : function
                    Error measurement function.

                Returns
                -------
                float
                    Fitness of the individual based on the evaluation.
                """

        # oversampling = torch.stack([self.tree.apply_tree(X_train[input_idx]) for input_idx in self.input_choice.repr_])
        oversampling = self.tree.apply_tree(X_train[self.input_choice.repr_])

        try:
            new_train_set = torch.concatenate((X_train, oversampling), axis = 0)

            # base_model = clone(base_model)

            base_model.fit(new_train_set, y_train_extended)
            self.predictions = torch.tensor(base_model.predict(X_test)).float()

            self.fitness = error_measure(y_test, self.predictions)

            self.f1_score = f1_score(y_test, self.predictions, average='weighted')
            self.recall = recall_score(y_test, self.predictions, average='weighted')
            self.precision = precision_score(y_test, self.predictions, average='weighted')
            self.gscore = geometric_mean_score(y_test, self.predictions, average='weighted')
            self.accuracy = accuracy_score(y_test, self.predictions)



        except:
            self.predictions = None

            self.fitness = 0

            self.f1_score = 0
            self.recall = 0
            self.precision = 0
            self.gscore = 0
            self.accuracy = 0

        # self.metrics = [self.f1_score, self.recall, self.precision, self.gscore, self.accuracy]

        return self.fitness

    def evaluate_test(self, base_model, X_train, y_train_extended, X_test, y_test, error_measure):
        """
                Evaluates the individual's test fitness using a base model and train/test datasets.

                Parameters
                ----------
                base_model : object
                    Base model for evaluating the individual.

                X_train : tensor
                    Training input data.

                y_train_extended : tensor
                    Extended training target data.

                X_test : tensor
                    Test input data.

                y_test : tensor
                    Test target data.

                error_measure : function
                    Error measurement function.
                """
        # oversampling = torch.stack([self.tree.apply_tree(X_train[input_idx]) for input_idx in self.input_choice.repr_])
        oversampling = self.tree.apply_tree(X_train[self.input_choice.repr_])

        new_train_set = torch.concatenate((X_train, oversampling), axis = 0)

        # base_model = clone(base_model)
        base_model.fit(new_train_set, y_train_extended)
        predictions = base_model.predict(X_test)

        self.test_pred = predictions

        self.test_fitness = error_measure(y_test, predictions)


        self.f1_score_test = f1_score(y_test, predictions, average='weighted')
        self.recall_test = recall_score(y_test, predictions, average='weighted')
        self.precision_test = precision_score(y_test, predictions, average='weighted')
        self.gscore_test = geometric_mean_score(y_test, predictions, average='weighted')
        self.accuracy_test = accuracy_score(y_test, predictions)

        # self.metrics_test = [self.f1_score_test, self.recall_test, self.precision_test, self.gscore_test, self.accuracy_test]

        # return self.fitness
