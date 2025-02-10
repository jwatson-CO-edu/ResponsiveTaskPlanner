########## INIT ####################################################################################

from uuid import uuid4
from copy import deepcopy



########## COMPONENTS ##############################################################################

class Value:
    """ A part of a symbol """
    def __init__( self, vType = "", value = None ):
        """ Create a null value with no type """
        self.typ : str = vType
        self.val       = value
        


class BKU:
    """ Basic Knowledge Unit: Can be a Node or an Edge """

    def pyval( self ):
        """ Return a version of this unit that code can operate on """
        raise NotImplementedError( f"NO METHOD to compute the value of {self.__class__.__name__}" )
    

    def ground( self ):
        """ Set unspecified values """
        raise NotImplementedError( f"NO METHOD to ground {self.__class__.__name__}" )
    

    def p_grounded( self ):
        """ Return whether all the `Value`s in the symbol have been grounded """
        if len( self.symbol ):
            for val_i in self.symbol:
                if val_i.val is None:
                    return False
            return True
        else:
            return False
    

    def make_symbol( self, valPairs ):
        """ Create a symbolic description of the `Node` """
        self.symbol = list()
        for pair in valPairs:
            pLen = len( pair )
            if pLen == 2:
                self.symbol.append( Value( pair[0], pair[1] ) )
            elif pLen == 1:
                self.symbol.append( Value( pair[0] ) )
            else:
                raise ValueError( f"CANNOT create a value from {pLen} items!: Got {pair}" )
            

    def symbol_hash( self ):
        """ Return a hashable version of the symbol """
        pairs = list()
        for val_i in self.symbol:
            pairs.append( (val_i.typ, (f"{val_i.val}" if (val_i.val is not None) else ""),) )
        return tuple( pairs )
    

    def set_val( self, vType, value ):
        """ Ground one of the elements of the symbol, Return whether a `Value` was set """
        for val_i in self.symbol:
            if val_i.typ == vType:
                val_i.val = deepcopy( value )
                return True
        return False


    def __init__( self ):
        """ Create an empty unit """
        self.id     : str       = str( uuid4() )
        self.symbol : list[Value] = list() # Discrete planning description
        self.prob   : float     = 0.0 # ---- Probability of the node/edge being true
        self.data   : dict      = dict() # - Generalized data store
        self.desc   : str       = "" # ----- Text desciption populated by ????
        self.edges  : list[BKU] = list() # - Knowledge Unit neighbors



class Node( BKU ):
    """ Container for beliefs/desires of all kinds """

    def __init__( self ):
        """ Create an empty node """
        super().__init__()
        self.dist : dict = dict() #- Distribution of symbols this could be
        


class Action( BKU ):
    """ Describes a probabilistic action """

    def __init__( self ):
        """ Create an empty action """
        super().__init__()
        self.preC : dict = dict() # Preconditions, and the probabilities that they are necessary for success
        self.pstC : dict = dict() # Postconditions, and the probabilities that they will become true
        self.exec        = None # - Executable code



########## BLOCKS PROBLEM ##########################################################################

class GraspObj( Node ):
    """ An object that can be grasped """

    def empty_symbol( self ):
        """ Ungrounded graspable object """
        self.make_symbol([ ( "class", "GraspObj", ), ( "name", ), ( "pose", ) ])


    def __init__( self ):
        """ Generic graspable object """
        super().__init__()
        self.empty_symbol()


    def ground( self, name, pose ):
        """ Specify a graspable obj in the scene """
        self.set_val( "name", name )
        self.set_val( "pose", pose )