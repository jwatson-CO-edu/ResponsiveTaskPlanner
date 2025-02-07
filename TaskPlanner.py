########## INIT ####################################################################################

from collections import deque

from py_trees.common import Status


########## TASK PLANNER ############################################################################


class Job:
    def __init__( self, name, orig, data ):
        self.name = name
        self.orig = orig
        self.data = data


class TaskAgent:
    """ The Method """

    def __init__( self ):
        """ Init planner """
        self.status = Status.INVALID
        self.q      = deque()


    def run_job( self, job ):
        pass


    def solve_task( self ):
        pass
    



