from gp4os.utils.utils_tree import create_grow_random_tree, create_full_random_tree
from gp4os.utils.utils_input_set import create_random_input_set
from gp4os.base.individual import Individual
# from gp4os.base.tree import Tree
# from gp4os.base.input_set import Input_Set

def grow(size, depth, FUNCTIONS, TERMINALS, CONSTANTS, input_set_size, umbalanced_obs_ind, p_c = 0.1):
    """
       Generates a list of individuals with random trees for a GM4OS population using the Grow method.

       Parameters
       ----------
       size : int
           The total number of individuals to be generated for the population.

       depth : int
           The maximum depth of the trees.

       FUNCTIONS : list
           The list of functions allowed in the trees.

       TERMINALS : list
           The list of terminal symbols allowed in the trees.

       CONSTANTS : list
           The list of constant values allowed in the trees.

       input_set_size : int
           The size of the input set for each individual.

       umbalanced_obs_ind : list
           The list of unbalanced observation indices used in creating the random input set.

       p_c : float, optional
           The probability of choosing a function node during tree creation. Default is 0.3.

       Returns
       -------
       list
           A list of Individual objects containing random trees and input sets based on the parameters provided.
       """

    return [Individual([create_grow_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c),
             create_random_input_set(input_set_size, umbalanced_obs_ind)], FUNCTIONS, TERMINALS, CONSTANTS) for _ in range(2, size+1) ]


def full(size, depth, FUNCTIONS, TERMINALS, CONSTANTS, input_set_size, umbalanced_obs_ind, p_c = 0.1):
    """
           Generates a list of individuals with random trees for a GM4OS population using the Full method.

           Parameters
           ----------
           size : int
               The total number of individuals to be generated for the population.

           depth : int
               The maximum depth of the trees.

           FUNCTIONS : list
               The list of functions allowed in the trees.

           TERMINALS : list
               The list of terminal symbols allowed in the trees.

           CONSTANTS : list
               The list of constant values allowed in the trees.

           input_set_size : int
               The size of the input set for each individual.

           umbalanced_obs_ind : list
               The list of unbalanced observation indices used in creating the random input set.

           p_c : float, optional
               The probability of choosing a function node during tree creation. Default is 0.3.

           Returns
           -------
           list
               A list of Individual objects containing random trees and input sets based on the parameters provided.
           """

    return [Individual([create_full_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c),
             create_random_input_set(input_set_size, umbalanced_obs_ind)], FUNCTIONS, TERMINALS, CONSTANTS) for _ in range(2, size+1)]

def rhh(size, depth, FUNCTIONS, TERMINALS, CONSTANTS, input_set_size, umbalanced_obs_ind, p_c = 0.1):
    """
           Generates a list of individuals with random trees for a GM4OS population using the ramped-half-half method.

           Parameters
           ----------
           size : int
               The total number of individuals to be generated for the population.

           depth : int
               The maximum depth of the trees.

           FUNCTIONS : list
               The list of functions allowed in the trees.

           TERMINALS : list
               The list of terminal symbols allowed in the trees.

           CONSTANTS : list
               The list of constant values allowed in the trees.

           input_set_size : int
               The size of the input set for each individual.

           umbalanced_obs_ind : list
               The list of unbalanced observation indices used in creating the random input set.

           p_c : float, optional
               The probability of choosing a function node during tree creation. Default is 0.3.

           Returns
           -------
           list
               A list of Individual objects containing random trees and input sets based on the parameters provided.
           """

    population = []

    inds_per_bin = size/(depth-1)
    for curr_depth in range(2, depth+1):
        population.extend([Individual([create_full_random_tree(curr_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c),
             create_random_input_set(input_set_size, umbalanced_obs_ind)], FUNCTIONS, TERMINALS, CONSTANTS) for _ in range(int(inds_per_bin//2))])
        population.extend([Individual([create_grow_random_tree(curr_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c),
             create_random_input_set(input_set_size, umbalanced_obs_ind)], FUNCTIONS, TERMINALS, CONSTANTS) for _ in range(int(inds_per_bin//2))])

    while len(population) < size:
        population.append(Individual([create_grow_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c),
                        create_random_input_set(input_set_size, umbalanced_obs_ind)], FUNCTIONS, TERMINALS, CONSTANTS))

    return population