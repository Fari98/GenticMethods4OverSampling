from GM4OS.utils.utils import flatten

class Population():

    """
    Represents a population of individuals in genetic programming.

    Attributes
    ----------
    pop : list
        List containing individuals in the population.

    size : int
        Number of individuals in the population.

    nodes_count : int
        Total count of nodes in all individuals' trees in the population.

    Methods
    -------
    __init__(pop)
        Initializes a Population object.

    evaluate(base_model, X_train, y_train_extended, X_test, y_test, error_measure)
        Evaluates the fitness of individuals in the population.
    """

    def __init__(self, pop):
        """
                Initializes a Population object.

                Parameters
                ----------
                pop : list
                    List containing individuals in the population.
        """

        self.pop = pop
        self.size = len(pop)

        self.nodes_count = sum([len(list(flatten(ind.tree_repr_))) for ind in pop])

    def evaluate(self, base_model, X_train, y_train_extended, X_test, y_test, error_measure):
        """
                Evaluates the fitness of individuals in the population using a base model and train/test datasets.

                Parameters
                ----------
                base_model : object
                    Base model for evaluating individuals.

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

        self.fit = [individual.evaluate(base_model, X_train, y_train_extended, X_test, y_test, error_measure)
                    for individual in self.pop]