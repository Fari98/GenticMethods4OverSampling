import random

def mutate_input_set(umbalanced_obs_ind, p_m = 0.5):

    """
    Mutates the input set based on a probability 'p_m' for each bit.

    Parameters
    ----------
    umbalanced_obs_ind : list
        The list of indices of the minority class.

    p_m : float, optional
        The probability of mutation for each bit in the input set. Default is 0.5.

    Returns
    -------
    function
        A function that applies mutation to an input set based on the defined probabilities.


    """

    def m_is(input_set):

            """
            Mutates the input set based on a probability 'p_m' for each bit.

            Parameters
            ----------
            input_set : list
                input_set to be mutated

            Returns
            -------
            list
                mutated input_set.


            """


            mutated_input_set = []

            for bit in input_set:

                if random.random() < p_m:

                    mutated_input_set.append(random.sample(umbalanced_obs_ind, k = 2))

                else:

                    mutated_input_set.append(bit)

            return mutated_input_set

    return m_is

def multiclass_mutate_input_set(indices_minority_classes, p_m = 0.5): #check wheter or not index needs to be + 1

    # indices_minority_classes collection of collections

    def m_is(input_set):

            mutated_input_set = []

            prev_index = 0

            for index, minority_class in indices_minority_classes:

                max_index = index + prev_index

                for i in range(prev_index, max_index):

                    if random.random() < p_m:

                        mutated_input_set.append(random.sample(minority_class, k = 2))

                    else:

                        mutated_input_set.append(input_set[i])

                prev_index = index

            return mutated_input_set

    return m_is


