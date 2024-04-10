class Input_Set():

    """
    Represents a set of inputs for the genetic programming individual.

    Attributes
    ----------
    repr_ : object
        Representation of the input set.

    Methods
    -------
    __init__(repr_)
        Initializes an Input_Set object.
    """

    def __init__(self, repr_):

        """
         Initializes an Input_Set object.

         Parameters
         ----------
         repr_ : object
             Representation of the input set.
         """

        self.repr_ = repr_