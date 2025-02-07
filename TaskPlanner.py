########## INIT ####################################################################################

from collections import deque

from Exec import TStat, StateExec


########## TASK PLANNER ############################################################################

class TaskAgent:
    """ The Method """

    def __init__( self, bgnMode : TStat ):
        """ Init planner """
        self.state                    = bgnMode
        self.rules  : list[StateExec] = list()
        self.status : TStat           = TStat.INVALID
        self.msgs   : deque           = deque()


    def run( self ):
        """ Run one step """

        ### Transition Rules ###

        ### Run Mode ###
        for rule in self.rules():
            rule.recv( self.state )
            if rule.status == TStat.SUCCESS:
                self.msgs.append( rule.send() )
