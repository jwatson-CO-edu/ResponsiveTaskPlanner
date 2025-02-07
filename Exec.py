########## INIT ####################################################################################

from enum import Enum
from collections import deque
from copy import deepcopy



########## ENUM ####################################################################################

class TStat( Enum ):
    """ What is the agent doing? """
    # INIT = -1
    ## State ##
    PERCEIVE =  0
    SYMB_PLN = 10
    MOTN_PLN = 20
    EXEC_ACT = 30
    ## Status ##
    INVALID = 100
    RUNNING = 200
    PAUSED  = 250
    FAILURE = 300
    SUCCESS = 400



########## HELPER CLASSES ##########################################################################

class Msg:
    """ Container class for data sent between states """
    def __init__( self, name : TStat, data ):
        """ Holds origin and payload """
        self.name : TStat = name
        self.data         = data




########## VIRTUAL STATE EXECUTOR ##################################################################

class StateExec:
    """ Manages activity for a state """

    def __init__( self, name : TStat, pauseAllowed : bool = False ):
        self.name      : TStat = name
        self.status    : TStat = TStat.INVALID
        self.p_canPaus : bool  = pauseAllowed
        self.data              = None
        self.history   : deque = deque()


    def p_running( self ):
        """ Is this state running? """
        return self.status == TStat.RUNNING
    

    def p_paused( self ):
        """ Is this state paused? """
        return self.status == TStat.PAUSED
    

    def p_done( self ):
        """ Has the state finished its work? """
        return self.status in (TStat.SUCCESS, TStat.FAILURE,)


    def begin( self ):
        """ Execute state activities """
        raise NotImplementedError( f"{self.__class__.__name__} needs `begin()` implementation!" )


    def tick( self ):
        """ Execute state activities """
        raise NotImplementedError( f"{self.__class__.__name__} needs `tick()` implementation!" )
    

    def end( self ):
        """ Shut down state activities """
        raise NotImplementedError( f"{self.__class__.__name__} needs `end()` implementation!" )
    

    def pause( self ):
        """ Pause state activities """
        self.status = TStat.PAUSED
    

    def resume( self ):
        """ Resume state activities following a pause """
        self.status = TStat.RUNNING


    def recv( self, desiredState : TStat, data = None ):
        """ Should this State `begin`/`end`? """
        # If our name was called, `resume` or `run`
        if desiredState == self.name:
            self.history.append( deepcopy( self.data ) )
            self.data = data
            if self.p_paused():
                self.resume()
            elif (not self.p_running()):
                self.tick()
        # Else someone else's turn, `pause` or `end`
        else:
            if self.p_canPaus and self.p_running():
                self.pause()
            else:
                self.end()


    def send( self ):
        """ Transmit the result of state activity """
        return Msg( self.name, self.data )




    