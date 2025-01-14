from GM4OS.utils.utils import bound_value
import torch

class Tree():
    """
        Represents a tree structure for genetic programming.

        Attributes
        ----------
        repr_ : object
            Representation of the tree structure.

        FUNCTIONS : dict
            Dictionary of allowed functions in the tree.

        TERMINALS : dict
            Dictionary of terminal symbols allowed in the tree.

        CONSTANTS : dict
            Dictionary of constant values allowed in the tree.

        depth : int
            Depth of the tree structure.

        Methods
        -------
        __init__(repr_, FUNCTIONS, TERMINALS, CONSTANTS)
            Initializes a Tree object.

        apply_tree(inputs)
            Evaluates the tree on input vectors x and y.

        print_tree_representation(indent="")
            Prints the tree representation with indentation.
        """

    def __init__(self, repr_, FUNCTIONS, TERMINALS, CONSTANTS):

        """
                Initializes a Tree object.

                Parameters
                ----------
                repr_ : object
                    Representation of the tree structure.

                FUNCTIONS : dict
                    Dictionary of allowed functions in the tree.

                TERMINALS : dict
                    Dictionary of terminal symbols allowed in the tree.

                CONSTANTS : dict
                    Dictionary of constant values allowed in the tree.
        """

        self.repr_ = repr_
        self.FUNCTIONS = FUNCTIONS
        self.TERMINALS = TERMINALS
        self.CONSTANTS = CONSTANTS
        self.depth = len(repr_)

        self.fitness = None


    # Function to evaluate a tree on input vectors x and y.
    def apply_tree(self, inputs):

        """
                Evaluates the tree on input vectors x and y.

                Parameters
                ----------
                inputs : tuple
                    Input vectors x and y.

                Returns
                -------
                float
                    Output of the evaluated tree.
        """

        # x, y = inputs[0], inputs[1]
        if isinstance(self.repr_, tuple):  # If it's a function node
            function_name = self.repr_[0]
            arity = self.FUNCTIONS[function_name]['arity']

            left_subtree, right_subtree = self.repr_[1], self.repr_[2]
            left_subtree = Tree(left_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
            left_result = left_subtree.apply_tree(inputs)

            if arity == 2:

                right_subtree = Tree(right_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)

                right_result = right_subtree.apply_tree(inputs)
                output = self.FUNCTIONS[function_name]['function'](left_result, right_result)
            else:

                output = self.FUNCTIONS[function_name]['function'](left_result)


        else:  # If it's a terminal node

            if self.repr_ in list(self.TERMINALS.keys()):
                output = self.TERMINALS[self.repr_](inputs)

            elif self.repr_ in list(self.CONSTANTS.keys()):
                output = self.CONSTANTS[self.repr_](inputs)

            else:
                raise ValueError(f"Unknown terminal or constant: {self.repr_}")

        return bound_value(output, -1e12, 1e12)


    def evaluate(self, base_models,
                 X_train_sets, y_train_extended_sets,
                 X_validation_sets, y_validation_sets,
                 error_measure, sample_input,
                 ):


        inputs = [sample_input(X_train) for X_train in X_train_sets]

        if self.fitness is None:
            augmented_sets = [torch.concatenate((X_train_sets[i], self.apply_tree(inputs[i])), axis = 0)
                              for i in range(len(inputs))]
            #loop over models and datasets and average the fitness
            performance = 0
            total_count = 0
            for i, augmented_set in enumerate(augmented_sets):
                for base_model in base_models:

                    base_model.fit(augmented_set, y_train_extended_sets[i])
                    predictions = base_model.predict(X_validation_sets[i])

                    performance += error_measure(y_validation_sets[i], predictions)
                    total_count += 1

            self.fitness = performance / total_count

    def evaluate_test(self, base_models,
                 X_train_sets, y_train_extended_sets,
                 X_test_sets, y_test_sets,
                 error_measure, sample_input,
                 ):


        inputs = [sample_input(X_train) for X_train in X_train_sets]

        if self.fitness is None:
            augmented_sets = [torch.concatenate((X_train_sets[i], self.apply_tree(inputs[i])), axis = 0)
                              for i in range(len(inputs))]
            #loop over models and datasets and average the fitness
            performance = 0
            total_count = 0
            for i, augmented_set in enumerate(augmented_sets):
                for base_model in base_models:

                    base_model.fit(augmented_set, y_train_extended_sets[i])
                    predictions = base_model.predict(X_test_sets[i])

                    performance += error_measure(y_test_sets[i], predictions)
                    total_count += 1

            self.test_fitness = performance / total_count

    def print_tree_representation(self, indent=""):

        """
                Prints the tree representation with indentation.

                Parameters
                ----------
                indent : str, optional
                    Indentation for tree structure representation.
        """

        if isinstance(self.repr_, tuple):  # If it's a function node
            function_name = self.repr_[0]

            print(indent + f"{function_name}(")
            if self.FUNCTIONS[function_name]['arity'] == 2:
                left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                left_subtree = Tree(left_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
                right_subtree = Tree(right_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
                left_subtree.print_tree_representation(indent + "  ")
                right_subtree.print_tree_representation(indent + "  ")
            else:
                left_subtree = self.repr_[1]
                left_subtree = Tree(left_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
                left_subtree.print_tree_representation(indent + "  ")
            print(indent + ")")
        else:  # If it's a terminal node
            print(indent + f"{self.repr_}")












