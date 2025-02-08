########## INIT ####################################################################################

from OWLv2_Segment import Perception_OWLv2, _QUERIES



########## PERCEPTION ##############################################################################

class Perception:
    """ Handle perception activities """

    def __init__( self ):
        """ Start camera + OWLv2 """
        self.owl = Perception_OWLv2()
        self.owl.start_vision()


    def observe( self ):
        """ Get observation + metadata """
        return self.owl.segment( _QUERIES )
    