########## INIT ####################################################################################

from collections import deque

# from BT import TStat, StateExec


########## MEMORY ##################################################################################




class StateMemory:
    """ Contains memories of all the beliefs over time """

    def reset_memory( self ):
        """ Erase memory components """
        self.scan : list[GraspObj]   = list()
        self.mult : bool             = False
        self.bMem : BayesMemory      = BayesMemory()
        self.symH : Dict[uuid4,ThinSymbol] = dict()

    def __init__( self ):