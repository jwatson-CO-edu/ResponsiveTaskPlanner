########## INIT ####################################################################################

from collections import deque

from py_trees.common import Status

from perception import Perception


########## TASK PLANNER ############################################################################


class TaskAgent:
    """ The Method """

    def __init__( self ):
        """ Init planner """
        self.status = Status.INVALID
        self.perc   = Perception()


    def solve_task( self ):
        pass
    



